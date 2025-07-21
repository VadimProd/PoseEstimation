import cv2
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


bones = [
    (5, 6),              # Shoulders (left_shoulder – right_shoulder)
    (5, 7), (7, 9),      # Left arm: shoulder -> elbow -> wrist
    (6, 8), (8, 10),     # Right arm: shoulder -> elbow -> wrist
    (11, 12),            # Hips (left_hip – right_hip)
    (11, 13), (13, 15),  # Left leg: hip -> knee -> ankle
    (12, 14), (14, 16),  # Right leg: hip -> knee -> ankle
    (0, 1), (1, 3),      # nose -> left eye -> left ear
    (0, 2), (2, 4),      # nose -> right eye -> right ear
    (3, 5),              # left ear -> left_shoulder
    (4, 6),              # right ear -> right_shoulder
    (11, 5), (12, 6)     # Hips -> shoulders
]

CHECKERBOARD = (6, 9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# square_size = 25.0

objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# objp *= square_size

# 
# Find chessboard 
# 

objpoints = []
imgpointsL = []
imgpointsR = []

imagesL = glob.glob("calibration_images/right*.jpg")
imagesR = glob.glob("calibration_images/left*.jpg")
image_sizeL = None
image_sizeR = None

for imageL, imageR in zip(imagesL, imagesR):
    imgL = cv2.imread(imageL)
    imgR = cv2.imread(imageR)

    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    retL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, None)
    retR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, None)

    if retR and retL == True:
        objpoints.append(objp)

        # Improves the accuracy of the corners found.
        cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
        cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)

        imgpointsL.append(cornersL)
        imgpointsR.append(cornersR)

        # Draw corners
        imgL = cv2.drawChessboardCorners(imgL, CHECKERBOARD, cornersL, retL)
        imgR = cv2.drawChessboardCorners(imgR, CHECKERBOARD, cornersR, retR)

        # cv2.imshow('img',imgL)
        # cv2.imshow('img',img)
        # cv2.waitKey(0)

        if image_sizeL is None:
            image_sizeL = grayL.shape[::-1]  # width, height

        if image_sizeR is None:
            image_sizeR = grayR.shape[::-1]  # width, height
            
# cv2.destroyAllWindows()

"""
cv2.calibrateCamera(...) -> [
    ret:    status (true/false)
    K:      calibration matrices for the camera
    dist:   distortion parameters
    rvecs:  rotation vector
    tvecs:  camera translation vector
]
"""
retL, KL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, image_sizeL, None, None)
retR, KR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, image_sizeR, None, None)

hL, wL, channelsL = imgL.shape
hR, wR, channelsR = imgR.shape

# Calculates optimal matrices for calibration to improve output quality.
KL, rvecsL = cv2.getOptimalNewCameraMatrix(KL, distL, (wL, hL), 1, (wL, hL))
KR, rvecsR = cv2.getOptimalNewCameraMatrix(KR, distR, (wR, hR), 1, (wR, hR))

print(f"KL = {KL}\nKR = {KR}")

# 
# Stereo calibration
# 

flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

print(f"criteria_stereo = {(criteria_stereo)}")

retStereo, _, _, _, _, R, t, E, F = cv2.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR, KL, distL, KR, distR, grayL.shape[::-1], criteria_stereo, flags
)

R = np.array([
    [0.29465049, 0.3249069, -0.89867491], 
    [0.29093463, 0.86528089, 0.408223], 
    [0.9102407, -0.38173876, 0.16042877]
])
E = np.array([
    [-6.48637009, -8.44611266, -5.18030757], 
    [-6.43190458, 7.76975398, -11.88506249], 
    [4.15546784, 10.42288535, 1.22386663]
])
F = np.array([
    [6.49311501e-06, 8.51722955e-06, -1.96058936e-03], 
    [6.46943376e-06, -7.87270641e-06, 5.13582685e-03], 
    [-5.64434634e-03, -5.77865050e-03, 1.00000000e+00]
])
t = np.array([
    [10.72730355], 
    [-3.5110198], 
    [11.31009029]
])

print(f"R = {R}\nE = {E}\nF = {F}")

# retStereo, KL, distL, KR, distR, R, t, E, F = cv2.stereoCalibrate(
#     objpoints, imgpointsL, imgpointsR, KL, distL, KR, distR, grayL.shape[::-1], criteria_stereo, flags
# )

# 
# Stereo rectification (?)
# 

rectifyScale = 1
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    KL, distL, KR, distR,
    image_sizeL, R, t,
    rectifyScale, (0,0)
)

print(f"RL = {R1}\nRR = {R2}")
print(f"T = {t}")

# Stereo maps for remapping

map1x, map1y = cv2.initUndistortRectifyMap(KL, distL, R1, P1, image_sizeL, cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(KR, distR, R2, P2, image_sizeR, cv2.CV_32FC1)

rectifiedL = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
rectifiedR = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)

#ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)
# print("K =", K)
# print("dist =", dist)

print("✅ Calibration complete!")

def normilize_coords(points3D):
    hips =  (points3D[0, 11, 1] + points3D[0, 12, 1])/2
    dif = np.abs(0.97 - hips)
    points3D[:, :, 1] += dif
    return points3D

def get3Dv5(pts1, pts2):
    # # Ректифицировать (нормализовать) точки
    # pointsL_norm = cv2.undistortPoints(np.expand_dims(pts1, axis=1), KL, distL, R=R1, P=P1)
    # pointsR_norm = cv2.undistortPoints(np.expand_dims(pts2, axis=1), KR, distR, R=R2, P=P2)

    # # Триангуляция
    # points4D = cv2.triangulatePoints(P1, P2, pointsL_norm, pointsR_norm)
    # points3D = (points4D[:3] / points4D[3]).T

    # Required format
    pts1 = np.array(pts1, dtype=np.float32).reshape(-1, 1, 2)
    pts2 = np.array(pts2, dtype=np.float32).reshape(-1, 1, 2)

    # Remove distorsion
    pts1_undist = cv2.undistortPoints(pts1, KL, distL)
    pts2_undist = cv2.undistortPoints(pts2, KR, distR)

    # Projection matrices
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))   # [I | 0]
    P2 = np.hstack((R, t))                          # [R | T]

    # Required format (for triangulatePoints)
    pts1_undist = pts1_undist.reshape(-1, 2).T  # (2, N)
    pts2_undist = pts2_undist.reshape(-1, 2).T

    # Triangulation
    points4D_hom = cv2.triangulatePoints(P1, P2, pts1_undist, pts2_undist)  # shape: (4, N)
    points3D = (points4D_hom[:3] / points4D_hom[3]).T  # shape: (N, 3)

    # Alignment relative to the pelvis
    pelvis = (points3D[11] + points3D[12])/2
    points3D -= pelvis

    # Z rotation 180 deg
    Rx = cv2.Rodrigues(np.array([0, 0, -np.pi]))[0]
    points3D = (Rx @ points3D.T).T

    return points3D

def draw3D(frames_3d):
    global bones

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        ax.cla()
        pts = frames_3d[frame]
        
        ax.set_xlim(-5, 5)
        ax.set_ylim(-6, 6)
        ax.set_zlim(-5, 5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"Frame {frame}")
        
        for (i, j) in bones:
            x = [pts[i, 0], pts[j, 0]]
            y = [pts[i, 1], pts[j, 1]]
            z = [pts[i, 2], pts[j, 2]]
            ax.plot(x, y, z, 'ro-')

        for idx, (x, y, z) in enumerate(pts):
            ax.text(x, y, z, str(idx), fontsize=8, color='blue')
           
        return []

    ani = FuncAnimation(fig, update, frames=len(frames_3d), blit=False)
    plt.show()

def draw2D(frames_2d):
    global bones
    # Format: [ [x0, y0, score0], [x1, y1, score1], ..., [x16, y16, score16] ]
    keypoints = frames_2d

    # Rendering
    plt.figure(figsize=(6, 8))
    for i, (x, y) in enumerate(keypoints):
        plt.scatter(x, y, c='red')
        plt.text(x + 3, y, str(i), fontsize=8)

    # Connecting bones
    for i, j in bones:
        x = [keypoints[i][0], keypoints[j][0]]
        y = [keypoints[i][1], keypoints[j][1]]
        plt.plot(x, y, 'b-', linewidth=2)

    plt.gca().invert_yaxis()
    plt.title("2D Skeletal Keypoints (COCO-style)")
    plt.axis('equal')
    plt.grid(True)
    plt.show()

with open('warmup_keypoints_left.json', 'r') as f:
    left_keypoints_json = json.load(f)

with open('warmup_keypoints_right.json', 'r') as f:
    right_keypoints_json = json.load(f)

frames_3d = []
start = 50
for i in range(start, len(left_keypoints_json)):#len(left_keypoints_json)):
    pts1 = np.array([np.float32(keypoint[:-1]) for keypoint in left_keypoints_json[i]], dtype=np.float32)
    pts2 = np.array([np.float32(keypoint[:-1]) for keypoint in right_keypoints_json[i]], dtype=np.float32)
    frames_3d.append(get3Dv5(pts1, pts2))

frames_3d = normilize_coords(np.array(frames_3d))

draw3D(frames_3d=frames_3d)
# print(frames_3d)

#draw2D(frames_2d=pts1)

frames_3d = [
    [point.tolist() for point in frame]
    for frame in frames_3d
]

# Save in json
with open("frames_3d.json", "w") as f:
    json.dump(frames_3d, f, indent=4)

# Save for animation in Unity
scale = 1

with open('My project (1)\\Assets\\pos_1.txt', 'w') as fout:
    for frame in frames_3d:
        line_parts = []
        coords = []
        for i, point in enumerate(frame):
            x_mm = point[0] * scale
            y_mm = point[1] * scale
            z_mm = point[2] * scale
            
            coords.append(np.array([x_mm, y_mm, z_mm]))

        coords = np.array(coords)
        hips = (coords[11] + coords[12])/2
        neck = (coords[5] + coords[6])/2
        spine = (hips + neck)/2

        coords_final = [
            hips,                                           # Hips          (0)
            spine,                                          # Spine         (1)
            coords[12], coords[14], coords[16],             # Right leg     (2, 3, 4)
            coords[11], coords[13], coords[15],             # Left leg      (5, 6, 7)
            (coords[5] + coords[6])/2,                      # Neck          (9)
            ((coords[3] + coords[4])/2 + coords[0])/2,      # Head          (10)
            coords[0],                                      # Nose
            coords[1], coords[2],                           # Eye
            coords[3], coords[4],                           # Ear
            coords[5], coords[7], coords[9],                # Left hand     (11, 12, 13)
            coords[6], coords[8], coords[10],               # Right hand    (14, 15, 16)
        ]

        for i, c in enumerate(coords_final):
            line_parts.append(f"{i} {c[0]:.3f} {c[1]:.3f} {c[2]:.3f},")

        fout.write(" ".join(line_parts) + "\n")