
from __future__ import print_function
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import time
import logging as log
import numpy as np
from openvino.inference_engine import IECore


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",
                      required=True, type=str)
    args.add_argument("-i", "--input",
                      help="Required. Path to video file or image. 'cam' for capturing video stream from camera",
                      required=True, type=str)
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. Absolute path to a shared library with the "
                           "kernels implementations.", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is "
                           "acceptable. The demo will look for a suitable plugin for device specified. "
                           "Default value is CPU", default="CPU", type=str)
    args.add_argument("-pt", "--prob_threshold", help="Optional. Probability threshold for detections filtering",
                      default=0.5, type=float)

    return parser


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    log.info("Creating Inference Engine...")
    ie = IECore()
    if args.cpu_extension and 'CPU' in args.device:
        ie.add_extension(args.cpu_extension, "CPU")
    # Read IR
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = ie.read_network(args.model, os.path.splitext(args.model)[0] + ".bin")



    img_info_input_blob = None
    feed_dict = {}
    for blob_name in net.inputs:
        if len(net.inputs[blob_name].shape) == 4:
            input_blob = blob_name
        elif len(net.inputs[blob_name].shape) == 2:
            img_info_input_blob = blob_name
        else:
            raise RuntimeError("Unsupported {}D input layer '{}'. Only 2D and 4D input layers are supported"
                               .format(len(net.inputs[blob_name].shape), blob_name))

    assert len(net.outputs) == 1, "Demo supports only single output topologies"

    out_blob = next(iter(net.outputs))
    log.info("Loading IR to the plugin...")
    exec_net = ie.load_network(network=net, num_requests=2, device_name=args.device)
    # Read and pre-process input image
    n, c, h, w = net.inputs[input_blob].shape
    if img_info_input_blob:
        feed_dict[img_info_input_blob] = [h, w, 1]

    if args.input == 'cam':
        input_stream = 0
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"

    cap = cv2.VideoCapture(input_stream)

    cur_request_id = 0
    next_request_id = 1

    log.info("Starting inference in async mode...")
    is_async_mode = True

    ret, frame = cap.read()
    
    disp=[]

    print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")

    COUNT=0
    
    while cap.isOpened():
        COUNT+=1
        if is_async_mode:
            ret, next_frame = cap.read()
        else:
            ret, frame = cap.read()
        if not ret:
            break
        initial_w = cap.get(3)
        initial_h = cap.get(4)

        inf_start = time.time()

        in_frame = cv2.resize(next_frame, (w, h))
        in_frame = in_frame.astype(np.float32)
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        in_frame = in_frame.reshape((n, c, h, w))            
        in_frame = in_frame / 255
        in_frame = np.expand_dims(in_frame, 0)
        feed_dict[input_blob] = in_frame
        exec_net.start_async(request_id=next_request_id, inputs=feed_dict)

        if exec_net.requests[cur_request_id].wait(-1) == 0:


            # Parse detection results of the current request
            #log.info("starting inference")
            res = exec_net.requests[cur_request_id].outputs[out_blob]

            #log.info("processing output blob")
            disp = res[0]
        
            # resize disp to input resolution
            disp = cv2.resize(disp, (int(initial_w), int(initial_h)), cv2.INTER_CUBIC)        
            # rescale disp
            disp_min = disp.min()
            disp_max = disp.max()
        
            if disp_max - disp_min > 1e-6:
                disp = (disp - disp_min) / (disp_max - disp_min)
            else:
                disp.fill(0.5)

            disp8=(disp*256).astype(np.uint8)
            heatmap = cv2.applyColorMap(disp8, cv2.COLORMAP_JET).astype(np.uint8)

            det_time = time.time() - inf_start
            
            inf_time = "Inference time: {} ms".format((int(det_time*1000)))
            fps = "Inference FPS: {} FPS".format(int(1/det_time))

            cv2.putText(heatmap, inf_time, (15, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (256, 256, 256), 1)
            cv2.putText(heatmap, fps, (15, 40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (256, 256, 256), 1)
           
            cv2.imshow("Detection Results", heatmap)
 
        cur_request_id, next_request_id = next_request_id, cur_request_id

        key = cv2.waitKey(1)
        if key == 27:
            break
        if (9 == key):
            is_async_mode = not is_async_mode
            log.info("Switched to {} mode".format("async" if is_async_mode else "sync"))

    cv2.destroyAllWindows()


if __name__ == '__main__':
    sys.exit(main() or 0)
