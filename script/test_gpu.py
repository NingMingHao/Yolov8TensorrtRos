
import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class YoloEngine:
    def __init__(self, model_path):
        self.engine = self.load_engine(model_path)
        print('engine loaded')
        self.context = self.engine.create_execution_context()
        self.input_binds, self.output_binds, self.allocations, self.outputs = self.allocate_buffers()

    def load_engine(self, model_path):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(model_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            print('read engine')
            return runtime.deserialize_cuda_engine(f.read())
        
    
    def allocate_buffers(self):
        input_binds = []
        output_binds = []
        allocations = []
        for binding in self.engine:
            binding_idx = self.engine.get_binding_index(binding)
            name = self.engine.get_binding_name(binding_idx) # Bindings correspond to the input and outputs of the network
            dtype = self.engine.get_binding_dtype(binding_idx)
            shape = self.engine.get_binding_shape(binding_idx)
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s

            allocation = cuda.mem_alloc(size)
            bind_description = {
                'index': binding_idx,
                'name': name,
                'dtype': np.dtype(trt.nptype(dtype)),
                'shape': list(shape),
                'allocation': allocation,
            }
            allocations.append(allocation)
            if self.engine.binding_is_input(binding):
                input_binds.append(bind_description)
            else:
                output_binds.append(bind_description)
        
            outputs = []
            for output in output_binds:
                outputs.append(np.empty(output['shape'], dtype=output['dtype']))
        return input_binds, output_binds, allocations, outputs

    def do_inference(self, context, bindings, inputs, outputs, stream):
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        stream.synchronize()
        return [out.host for out in outputs]
    
    def __call__(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        img = np.transpose(img, (2, 0, 1)).astype(np.float16) #for half precision
        img /= 255.0
        img = np.expand_dims(img, axis=0)
        img = np.ascontiguousarray(img)

        ## reference
        cuda.memcpy_htod(self.input_binds[0]['allocation'], img)
        self.context.execute_v2(self.allocations)
        for o in range(len(self.outputs)):
            cuda.memcpy_dtoh(self.outputs[o], self.output_binds[o]['allocation'])
        print(self.outputs)


class YoloRos:
    def __init__(self, model_path):
        self.yolo_engine = YoloEngine(model_path)
        print('yolo engine initialized')
        self.bridge = CvBridge()
        self.sub = rospy.Subscriber('/pylon_camera_node_left/image_rect/compressed', CompressedImage, self.callback, queue_size=1)
        self.pub = rospy.Publisher('/yolo/image_raw', Image, queue_size=1)
        

    def callback(self, msg):
        # compressed image to cv2 image
        np_arr = np.fromstring(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.yolo_engine(img)
        # det = self.yolo(img)
        # all_boxes = det[0].boxes
        # for ibox in all_boxes:
        #     ixyxy = ibox.xyxy.cpu().numpy()[0]
        #     cv2.rectangle(img, (int(ixyxy[0]), int(ixyxy[1])), (int(ixyxy[2]), int(ixyxy[3])), (0, 255, 0), 2)

        # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('img', 640, 480)
        # cv2.imshow('img', img)
        # cv2.waitKey(1)

if __name__ == '__main__':
    rospy.init_node('yolo_ros_node')
    # model_path = 'yolov8s.pt'
    model_path = '/home/jetson/Documents/test_yolov8/DeepStream-Yolo/model_b1_gpu0_fp32.engine'
    yolo_ros = YoloRos(model_path)
    rospy.spin()