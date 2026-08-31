# Copyright (c) Facebook, Inc. and its affiliates.
import atexit  # 用来在程序结束前注册清理函数
import bisect  # 有序列表插入工具，这里用来按顺序插入结果
import multiprocessing as mp  # 多进程模块，用于异步预测
from collections import deque  # 双端队列，用作缓冲区
import cv2  # OpenCV，用于读写视频/图像与颜色转换
import torch  # PyTorch 张量和设备管理

from detectron2.data import MetadataCatalog  # 用于获取数据集的元信息（类别、颜色等）
from detectron2.engine.defaults import DefaultPredictor  # Detectron2 的封装好的预测器
from detectron2.utils.video_visualizer import VideoVisualizer  # 用于视频结果可视化
from detectron2.utils.visualizer import ColorMode, Visualizer  # 用于图像结果可视化


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode): Detectron2 的配置对象
            instance_mode (ColorMode): 可视化时的颜色模式
            parallel (bool): 是否使用多进程异步预测（可加速视频处理）
        """
        # 从配置中取得测试集名称, 再从 MetadataCatalog 里获取该数据集的元信息
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        # 指定一个 CPU 设备，用于把预测结果转到 CPU 上做可视化
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        # 是否启用并行模式
        self.parallel = parallel
        if parallel:
            # 获取 GPU 数量
            num_gpu = torch.cuda.device_count()
            # 使用自定义的 AsyncPredictor，在多个 GPU 上异步运行模型
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            # 使用 Detectron2 自带的同步预测器
            self.predictor = DefaultPredictor(cfg)

        # 从配置中读取测试时的分数阈值；这里单独取出来做一个“小修补”
        self.threshold = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST  # workaround

    def run_on_image(self, image):
        """
        对一张图像做推理并可视化。

        Args:
            image (np.ndarray): 一张 OpenCV BGR 格式的图像, 形状为 (H, W, C)

        Returns:
            predictions (dict): 模型输出的预测结果（instances 等）
            vis_output (VisImage): 可视化后的图像对象
        """
        vis_output = None
        # 调用 predictor 得到预测结果
        predictions = self.predictor(image)
        # 从预测结果中取出实例信息
        instances = predictions['instances']
        # 根据分数阈值过滤掉低置信度目标
        new_instances = instances[instances.scores > self.threshold]
        # 构造新的预测字典，只保留过滤后的 instances
        predictions = {'instances': new_instances}
        # OpenCV 默认是 BGR，这里转为 RGB 方便 Visualizer 使用
        image = image[:, :, ::-1]
        # 构造一个可视化器，传入图像、元信息和实例显示模式
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        # 如果预测结果中有 panoptic 分割（全景分割）
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            # 绘制全景分割结果
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            # 如果有语义分割结果
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            # 如果有实例分割/检测结果
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                # 绘制实例预测（框、mask、类别等）
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        # 返回过滤后的预测和可视化结果
        return predictions, vis_output

    def _frame_from_video(self, video):
        """
        一个生成器：不断从视频中读取帧。
        """
        while video.isOpened():
            success, frame = video.read()
            if success:
                # 读到一帧就 yield 出去
                yield frame
            else:
                # 读不到说明结束
                break

    def run_on_video(self, video):
        """
        对视频的每一帧做预测并可视化。

        Args:
            video (cv2.VideoCapture): 一个 VideoCapture 对象，可以是摄像头或视频文件

        Yields:
            ndarray: 可视化后的每一帧，BGR 格式
        """
        # 针对视频的可视化器（与静态图稍有不同）
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            """
            把一帧图像和对应预测结果进行可视化，并返回 BGR 格式的图像。
            """
            # 把 BGR 转为 RGB 供 VideoVisualizer 使用
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            elif "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
            elif "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )

            # VideoVisualizer 返回的是 RGB，这里再转回 BGR 以便 OpenCV 写视频
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        # 获取一个逐帧生成器
        frame_gen = self._frame_from_video(video)
        if self.parallel:
            # 若并行，缓冲区大小由 AsyncPredictor 给定（默认 5 * num_procs）
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()  # 存放未处理/待取出的帧

            # 遍历视频帧
            for cnt, frame in enumerate(frame_gen):
                # 把帧存入队列
                frame_data.append(frame)
                # 把帧送到异步预测器中
                self.predictor.put(frame)

                # 当超过缓冲区大小时，就开始取预测结果并可视化
                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions)

            # 视频读完之后，可能还有未处理的帧和预测结果，继续取完
            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions)
        else:
            # 不并行，逐帧同步预测和可视化
            for frame in frame_gen:
                yield process_predictions(frame, self.predictor(frame))


class AsyncPredictor:
    """
    一个异步预测器，可以在多个 GPU 上并行运行模型。
    当可视化本身较耗时时（例如视频），使用它可以提高吞吐量。
    """

    class _StopToken:
        """
        特殊标记，用来通知子进程退出。
        """
        pass

    class _PredictWorker(mp.Process):
        """
        实际执行预测工作的子进程类。
        """

        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg  # 每个子进程持有一份配置
            self.task_queue = task_queue  # 主进程往这里丢任务
            self.result_queue = result_queue  # 子进程把结果放在这里
            super().__init__()

        def run(self):
            # 在子进程内创建 DefaultPredictor（模型实例）
            predictor = DefaultPredictor(self.cfg)

            while True:
                # 从任务队列中取任务
                task = self.task_queue.get()
                # 如果是停止标记，则退出循环结束进程
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                # 正常任务：task = (idx, data)
                idx, data = task
                # 运行预测
                result = predictor(data)
                # 把（索引, 预测结果）丢到结果队列中
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode): 配置
            num_gpus (int): GPU 数量。若为 0 则在 CPU 上跑。
        """
        # 使用的 worker 数量至少为 1（如果没有 GPU就用 CPU）
        num_workers = max(num_gpus, 1)
        # 任务队列和结果队列，最大长度是 worker 数量的 3 倍
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []  # 存放所有子进程

        # 为每个 GPU 创建一个子进程
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()  # 每个进程都要一份独立的 cfg
            cfg.defrost()  # 解除“冻结”，允许修改配置
            # 设置该进程使用的设备
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            # 创建并保存子进程
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        # put_idx: 已经放入任务的数量
        # get_idx: 已经取出的结果数量
        self.put_idx = 0
        self.get_idx = 0
        # result_rank: 已经收到但还没按顺序返回的结果索引
        # result_data: 与 result_rank 对应的结果
        self.result_rank = []
        self.result_data = []

        # 启动所有子进程
        for p in self.procs:
            p.start()
        # 注册退出时的清理函数（关闭子进程）
        atexit.register(self.shutdown)

    def put(self, image):
        """
        把一个图像任务丢进队列，等待子进程处理。
        """
        self.put_idx += 1
        # 任务是 (索引, 数据) 的形式
        self.task_queue.put((self.put_idx, image))

    def get(self):
        """
        从结果队列中按顺序获取预测结果。
        即使子进程返回是乱序的，这里也会排序后再返回。
        """
        # 当前需要的结果索引
        self.get_idx += 1
        # 如果 result_rank 有缓存，并且最前面的就是我们要的索引
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            # 从缓存中删除这条结果
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # 不断从结果队列中拿结果
            idx, res = self.result_queue.get()
            # 如果正好是我们要的那一条，直接返回
            if idx == self.get_idx:
                return res
            # 否则，把它插入到有序列表 result_rank 中（保持结果按 idx 排序）
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        """
        返回当前还未获取结果的任务数量 = 已提交任务数 - 已取出结果数
        """
        return self.put_idx - self.get_idx

    def __call__(self, image):
        """
        让 AsyncPredictor 像普通 predictor 一样可被直接调用：
        先 put 再 get。
        """
        self.put(image)
        return self.get()

    def shutdown(self):
        """
        关闭所有子进程：向每个进程发送停止标记。
        """
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        """
        默认缓冲区大小：每个进程 5 帧。
        视频推理时会用到。
        """
        return len(self.procs) * 5
