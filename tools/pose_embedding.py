import cv2
import numpy as np
import os
# from matplotlib import pyplot as plt
import time
import mediapipe as mp



mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities


def mediapipe_detection(image, model):
    '''
    Call mediapipe to process data
    ----------------
    input：
        image：BGR格式三维array
        model：此处为holistic  
    
    intermediate：
        results：landmark { x,y,z,visibility:float}
            
            face_landmarks, pose_landmarks, pose_world_landmarks,left/right_hand_landmarks大类的landmars list
            x, y, [z]：基于图片width和height    normolize至[0.0,1.0]的x, y, [z]轴坐标
            visibility: 该点可能被标定显示的likelihood

    return：
        image：3d array of RGB image
        results： lanmarks list   ： landmark { x,y,z,visibility:float}
    '''
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)             # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results


def draw_landmarks(image, results):
    '''
    基于landmarks绘制face，position，hands连线
    '''
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections


def draw_styled_landmarks(image, results):
    '''
    基于landmarks绘制face，position，hands的点
    '''
    # Draw face connections
    '''mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) '''
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 


def extract_keypoints(results):
    ''' 
    提取关键点信息
    以pose为例：识别到pose，则返回其result信息，否则补0000
    pose：shape（132, ) 
     '''
    embedder = FullBodyPoseEmbedder(train_flag=False)
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark],dtype="float32") if results.pose_landmarks else np.zeros([33,3])
    pose = embedder(pose)
    #face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    #lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    #rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose])


colors = [(245,117,16), (117,245,16), (16,117,245),(220,220,220),(250,128,114),(255,160,122),(255,69,0),(255,140,0),(255,165,0),(255,215,0),(184,134,11)]
def prob_viz(res, actions, input_frame, colors=colors):
    output_frame = input_frame.copy()
    res[np.isnan(res).astype(int)] = 0
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
        
    return output_frame

"""# Body pose Encoder"""


class FullBodyPoseEmbedder(object):
  """Converts 3D pose landmarks into 3D embedding."""

  def __init__(self, torso_size_multiplier=2.5,train_flag=True):
    # Multiplier to apply to the torso to get minimal body size.
    self._torso_size_multiplier = torso_size_multiplier
    self._train_flag = train_flag


    # Names of the landmarks as they appear in the prediction.
    self._landmark_names = [
                'hip', 
                'left_hip', 'left_knee', 'left_foot', 
                'right_hip', 'right_knee','right_foot',
                'spine', 'thorax', 'nose', 'head',
                'right_shoulder', 'right_elbow', 'right_wrist', 
                'left_shoulder', 'left_elbow', 'left_wrist' 
                
            ]

  def __call__(self, landmarks):
    """Normalizes pose landmarks and converts to embedding
    
    Args:
      landmarks - NumPy array with 3D landmarks of shape (N, 3).

    Result:
      Numpy array with pose embedding of shape (M, 3) where `M` is the number of
      pairwise distances defined in `_get_pose_distance_embedding`.
    """
    # assert landmarks.shape[0] == len(self._landmark_names), 'Unexpected number of landmarks: {}'.format(landmarks.shape[0])

    # Get pose landmarks.
    landmarks = np.copy(landmarks)
    # transform mediapipe into 17 joint points
    if self._train_flag == False:
        hip = (landmarks[23,:]+landmarks[24,:])/2
        landmarks = np.array([hip, #hip
                    landmarks[23,:],landmarks[25,:],landmarks[29,:], #hip, knee foot
                    landmarks[24,:],landmarks[26,:],landmarks[30,:],
                    (landmarks[0,:]-hip)*0.3+hip,(landmarks[0,:]-hip)*0.6+hip,landmarks[0,:],(landmarks[2,:]+landmarks[5,:])/2,
                    landmarks[12,:],landmarks[14,:],landmarks[16,:], ## shoulder elbow wrist
                    landmarks[11,:],landmarks[13,:],landmarks[15,:]
                      ])

    # Normalize landmarks.
    landmarks = self._normalize_pose_landmarks(landmarks)

    # Get embedding.
    embedding = self._get_pose_distance_embedding(landmarks)
    

    #return embedding
    return embedding

  def _normalize_pose_landmarks(self, landmarks):
    """Normalizes landmarks translation and scale."""
    landmarks = np.copy(landmarks)

    # Normalize translation.
    pose_center = self._get_pose_center(landmarks)
    landmarks -= pose_center

    # Normalize scale.
    pose_size = self._get_pose_size(landmarks, self._torso_size_multiplier)
    landmarks /= pose_size
    # Multiplication by 100 is not required, but makes it eaasier to debug.
    landmarks *= 100

    return landmarks

  def _get_pose_center(self, landmarks):
    """Calculates pose center as point between hips."""
    return landmarks[self._landmark_names.index('hip')]

  def _get_pose_size(self, landmarks, torso_size_multiplier):
    """Calculates pose size.
    
    It is the maximum of two values:
      * Torso size multiplied by `torso_size_multiplier`
      * Maximum distance from pose center to any pose landmark
    """
    # This approach uses only 2D landmarks to compute pose size.
    landmarks = landmarks[:, :2]

    # Hips center.
    hips = self._get_pose_center(landmarks)

    # Shoulders center.
    left_shoulder = landmarks[self._landmark_names.index('left_shoulder')]
    right_shoulder = landmarks[self._landmark_names.index('right_shoulder')]
    shoulders = (left_shoulder + right_shoulder) * 0.5

    # Torso size as the minimum body size.
    torso_size = np.linalg.norm(shoulders - hips)

    # Max dist to pose center.
    pose_center = self._get_pose_center(landmarks)
    max_dist = np.max(np.linalg.norm(landmarks - pose_center, axis=1))

    return max(torso_size * torso_size_multiplier, max_dist)

  def _calculate_angle(self, first, mid, end):  
    """
    Finds angle between three points.
    Args:
            first (_type_): _description_
            mid (_type_): _description_
            end (_type_): _description_

    Returns:
            _type_: _description_
    """
    a = np.array(first)  # First
    b = np.array(mid)  # Mid
    c = np.array(end)  # End
    radians_z = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    radians_y = np.arctan2(c[2] - b[2], c[0] - b[0]) - np.arctan2(a[2] - b[2], a[0] - b[0])
    radians_x = np.arctan2(c[2] - b[2], c[1] - b[1]) - np.arctan2(a[2] - b[2], a[1] - b[1])
    radians = np.array([radians_x,radians_y,radians_z])
    angle = np.abs(radians * 180.0 / np.pi)
    f = np.vectorize(lambda x: (360-x) if x>180 else x)
    return f(angle)    

  def _get_pose_distance_embedding(self, landmarks):
    """Converts pose landmarks into 3D embedding.

    We use several pairwise 3D distances to form pose embedding. All distances
    include X and Y components with sign. We differnt types of pairs to cover
    different pose classes. Feel free to remove some or add new.
    
    Args:
      landmarks - NumPy array with 3D landmarks of shape (N, 3).

    Result:
      Numpy array with pose embedding of shape (M, 3) where `M` is the number of
      pairwise distances.
    """
    # dists = np.array([
    #     # One joint.

    #     self._get_distance(
    #         self._get_average_by_names(landmarks, 'left_hip', 'right_hip'),
    #         self._get_average_by_names(landmarks, 'left_shoulder', 'right_shoulder')),

    #     self._get_distance_by_names(landmarks, 'left_shoulder', 'left_elbow'),
    #     self._get_distance_by_names(landmarks, 'right_shoulder', 'right_elbow'),

    #     self._get_distance_by_names(landmarks, 'left_elbow', 'left_wrist'),
    #     self._get_distance_by_names(landmarks, 'right_elbow', 'right_wrist'),

    #     self._get_distance_by_names(landmarks, 'left_hip', 'left_knee'),
    #     self._get_distance_by_names(landmarks, 'right_hip', 'right_knee'),

    #     self._get_distance_by_names(landmarks, 'left_knee', 'left_foot'),
    #     self._get_distance_by_names(landmarks, 'right_knee', 'right_foot'),

    #     # Two joints.

    #     self._get_distance_by_names(landmarks, 'left_shoulder', 'left_wrist'),
    #     self._get_distance_by_names(landmarks, 'right_shoulder', 'right_wrist'),

    #     self._get_distance_by_names(landmarks, 'left_hip', 'left_foot'),
    #     self._get_distance_by_names(landmarks, 'right_hip', 'right_foot'),

    #     # Four joints.

    #     self._get_distance_by_names(landmarks, 'left_hip', 'left_wrist'),
    #     self._get_distance_by_names(landmarks, 'right_hip', 'right_wrist'),

    #     # Five joints.

    #     self._get_distance_by_names(landmarks, 'left_shoulder', 'left_foot'),
    #     self._get_distance_by_names(landmarks, 'right_shoulder', 'right_foot'),
        
    #     self._get_distance_by_names(landmarks, 'left_hip', 'left_wrist'),
    #     self._get_distance_by_names(landmarks, 'right_hip', 'right_wrist'),

    #     # Cross body.

    #     self._get_distance_by_names(landmarks, 'left_elbow', 'right_elbow'),
    #     self._get_distance_by_names(landmarks, 'left_knee', 'right_knee'),

    #     self._get_distance_by_names(landmarks, 'left_wrist', 'right_wrist'),
    #     self._get_distance_by_names(landmarks, 'left_foot', 'right_foot'),

    #     # Body bent direction.

    #     # self._get_distance(
    #     #     self._get_average_by_names(landmarks, 'left_wrist', 'left_ankle'),
    #     #     landmarks[self._landmark_names.index('left_hip')]),
    #     # self._get_distance(
    #     #     self._get_average_by_names(landmarks, 'right_wrist', 'right_ankle'),
    #     #     landmarks[self._landmark_names.index('right_hip')]),
    # ])

     # add angles
    angles = np.array([
            # Angle
            # Siku kanan dan kiri (shoulder-elbow-wrist)
            self._get_angle_by_names(landmarks, 'left_shoulder', 'left_elbow', 'left_wrist'),
            self._get_angle_by_names(landmarks, 'right_shoulder', 'right_elbow', 'right_wrist'),

            # #sudut badan atas (shoulder-hip-knee)
            self._get_angle_by_names(landmarks, 'left_shoulder', 'left_hip', 'left_knee'),
            self._get_angle_by_names(landmarks, 'right_shoulder', 'right_hip', 'right_knee'),

            # #sudut badan bawah (hip-knee-foot)
            self._get_angle_by_names(landmarks, 'left_hip', 'left_knee', 'left_foot'),
            self._get_angle_by_names(landmarks, 'right_hip', 'right_knee', 'right_foot'),

            # #sudut ketek (elbow-shoulder-hip)
            self._get_angle_by_names(landmarks, 'left_elbow', 'left_shoulder', 'left_hip'),
            self._get_angle_by_names(landmarks, 'right_elbow', 'right_shoulder', 'right_hip'),

        ])

    # completes embedding
    # embedding = np.append(landmarks, dists, axis=0)
    embedding = np.append(landmarks, angles, axis=0)
    return embedding

  def _get_average_by_names(self, landmarks, name_from, name_to):
    lmk_from = landmarks[self._landmark_names.index(name_from)]
    lmk_to = landmarks[self._landmark_names.index(name_to)]
    return (lmk_from + lmk_to) * 0.5

  def _get_distance_by_names(self, landmarks, name_from, name_to):
    lmk_from = landmarks[self._landmark_names.index(name_from)]
    lmk_to = landmarks[self._landmark_names.index(name_to)]
    return self._get_distance(lmk_from, lmk_to)

  def _get_distance(self, lmk_from, lmk_to):
    return lmk_to - lmk_from
  def _get_angle_by_names(self, landmarks, name_first, name_mid, name_end):
    lmk_first = landmarks[self._landmark_names.index(name_first)]
    lmk_mid = landmarks[self._landmark_names.index(name_mid)]
    lmk_end = landmarks[self._landmark_names.index(name_end)]
    return self._calculate_angle(lmk_first, lmk_mid, lmk_end)