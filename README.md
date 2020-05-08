# Gaze Estimation 2D Demo with Sparking Laser Beam ;-)
This program demonstrates how to use the [`gaze-estimation-adas-0002`](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html) model of the OpenVINO [Open Model Zoo](https://docs.openvinotoolkit.org/latest/_models_intel_index.html) with [Intel(r) Distribution of OpenVINO(tm) toolkit](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html).  
This program finds the faces in an image, detect the landmark points on the detected faces to find the eyes, estimate the head rotation angle, and estimates the gaze orientation.  
This program draws the gaze lines like the laser beams. Also the program detects the collision of the laser beams and draws sparkles at the crossing point of the laser beams (for fun).
The gaze estimation model requires the head rotation angle and the cropped eye images as the input of the model. Therefore, the program uses [`head-pose-estimation-adas-0001`](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html) model to detect the head rotation angles and [`facial-landmarks-35-adas-0002`](https://docs.openvinotoolkit.org/latest/_models_intel_facial_landmarks_35_adas_0002_description_facial_landmarks_35_adas_0002.html) model to detects key landmark points on the face. The landmark detection model detects 35 points from a face.  

このプログラムは[Intel(r) Distribution of OpenVINO(tm) toolkit](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html)を使った、OpenVINO [Open Model Zoo](https://docs.openvinotoolkit.org/latest/_models_intel_index.html)の[`gaze-estimation-adas-0002`](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)(視線推定)モデルの使い方を示すためのデモプログラムです。  
プログラムはまず入力画像から顔を検出し、その後顔のランドマークポイントを検出し、頭の回転角度を検出し、最後に視線を推定します。  
プログラムはレーザービームのように視線を描画します。また、レーザービーム同士が交差した場合、そこにスパークを描画します(遊びです)。  
視線推定モデルは入力として頭の回転角度と切り抜いた２つの目の画像を必要とします。そのため、プログラムは[`head-pose-estimation-adas-0001`](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)モデルを使用して頭の回転角を推定し、[`facial-landmarks-35-adas-0002`](https://docs.openvinotoolkit.org/latest/_models_intel_facial_landmarks_35_adas_0002_description_facial_landmarks_35_adas_0002.html)モデルで顔のキーランドマークポイント（目や鼻の位置など）を推定しています。ランドマークモデルは１つの顔から35点のキーポイントを検出します。    

### Gaze Estimation Result
![gaze](./resources/gaze.gif)


### Required DL Models to Run This Demo

The demo expects the following models in the Intermediate Representation (IR) format:

  * `face-detection-adas-0001`
  * `head-pose-estimation-adas-0001`
  * `facial-landmarks-35-adas-0002`
  * `gaze-estimation-adas-0002`

You can download these models from OpenVINO [Open Model Zoo](https://github.com/opencv/open_model_zoo).
In the `models.lst` is the list of appropriate models for this demo that can be obtained via `Model downloader`.
Please see more information about `Model downloader` [here](../../../tools/downloader/README.md).

## How to Run


### 0. Prerequisites
- **OpenVINO 2020.2**
  - If you haven't installed it, go to the OpenVINO web page and follow the [*Get Started*](https://software.intel.com/en-us/openvino-toolkit/documentation/get-started) guide to do it.  

### 1. Install dependencies  
The demo depends on:
- `numpy`
- `scipy`
- `opencv-python`

To install all the required Python modules you can use:

``` sh
(Linux) pip3 install -r requirements.txt
(Win10) pip install -r requirements.txt
```

### 2. Download DL models from OMZ
Use `Model Downloader` to download the required models.
``` sh
(Linux) python3 $INTEL_OPENVINO_DIR/deployment_tools/tools/model_downloader/downloader.py --list models.lst
(Win10) python "%INTEL_OPENVINO_DIR%\deployment_tools\tools\model_downloader\downloader.py" --list models.lst
```

### 3. Run the demo app
Attach a USB webCam as input of the demo program, then run the program. If you want to use a movie file as an input, you can modify the source code to do it.  

*Following keys are valid:*  
`'f'`: Flip image  
`'l'`: Laser mode on/off  
`'s'`: Spark mode on/off  
`'b'`: Boundary box on/off  

``` sh
(Linux) python3 gaze-estimation.py
(Win10) python gaze-estimation.py
```

## Demo Output  
The application draws the results on the input image.

## Tested Environment  
- Windows 10 x64 1909 and Ubuntu 18.04 LTS  
- Intel(r) Distribution of OpenVINO(tm) toolkit 2020.2  
- Python 3.6.5 x64  

## See Also  
* [Using Open Model Zoo demos](../../README.md)  
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)  
* [Model Downloader](../../../tools/downloader/README.md)  
