# Copyright (C) 2023 Miguel Ángel González Santamarta

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import cv2
import numpy as np
from typing import List, Tuple
import os
import time
import rclpy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState
from scipy.spatial.transform import Rotation as Rot
import message_filters
from cv_bridge import CvBridge
from tf2_ros.buffer import Buffer
from tf2_ros import TransformException
from tf2_ros.transform_listener import TransformListener

from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import TransformStamped
from yolo_msgs.msg import Detection
from yolo_msgs.msg import DetectionArray
from yolo_msgs.msg import KeyPoint3D
from yolo_msgs.msg import KeyPoint3DArray
from yolo_msgs.msg import BoundingBox3D


class Detect3DNode(LifecycleNode):

    def __init__(self) -> None:
        super().__init__("bbox3d_node")

        # parameters
        self.declare_parameter("min_seg_points_for_orientation", 20)
        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("maximum_detection_threshold", 0.3)
        self.declare_parameter("depth_image_units_divisor", 1000)
        self.declare_parameter(
            "depth_image_reliability", QoSReliabilityPolicy.BEST_EFFORT
        )
        self.declare_parameter("depth_info_reliability", QoSReliabilityPolicy.BEST_EFFORT)

        # aux
        self.tf_buffer = Buffer()
        self.cv_bridge = CvBridge()

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Configuring...")

        self.min_seg_points_for_orientation = (
            self.get_parameter("min_seg_points_for_orientation")
            .get_parameter_value()
            .integer_value
        )
        self.target_frame = (
            self.get_parameter("target_frame").get_parameter_value().string_value
        )
        self.maximum_detection_threshold = (
            self.get_parameter("maximum_detection_threshold")
            .get_parameter_value()
            .double_value
        )
        self.depth_image_units_divisor = (
            self.get_parameter("depth_image_units_divisor")
            .get_parameter_value()
            .integer_value
        )
        dimg_reliability = (
            self.get_parameter("depth_image_reliability")
            .get_parameter_value()
            .integer_value
        )

        self.depth_image_qos_profile = QoSProfile(
            reliability=dimg_reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        dinfo_reliability = (
            self.get_parameter("depth_info_reliability")
            .get_parameter_value()
            .integer_value
        )

        self.depth_info_qos_profile = QoSProfile(
            reliability=dinfo_reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # pubs
        self._pub = self.create_publisher(DetectionArray, "detections_3d", 10)

        super().on_configure(state)
        self.get_logger().info(f"[{self.get_name()}] Configured")

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Activating...")

        # subs
        self.depth_sub = message_filters.Subscriber(
            self, Image, "depth_image", qos_profile=self.depth_image_qos_profile
        )
        self.depth_info_sub = message_filters.Subscriber(
            self, CameraInfo, "depth_info", qos_profile=self.depth_info_qos_profile
        )
        self.detections_sub = message_filters.Subscriber(
            self, DetectionArray, "detections"
        )

        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            (self.depth_sub, self.depth_info_sub, self.detections_sub), 10, 0.5
        )
        self._synchronizer.registerCallback(self.on_detections)

        super().on_activate(state)
        self.get_logger().info(f"[{self.get_name()}] Activated")

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Deactivating...")

        self.destroy_subscription(self.depth_sub.sub)
        self.destroy_subscription(self.depth_info_sub.sub)
        self.destroy_subscription(self.detections_sub.sub)

        del self._synchronizer

        super().on_deactivate(state)
        self.get_logger().info(f"[{self.get_name()}] Deactivated")

        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Cleaning up...")

        del self.tf_listener

        self.destroy_publisher(self._pub)

        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Cleaned up")

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Shutting down...")
        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Shutted down")
        return TransitionCallbackReturn.SUCCESS

    def on_detections(
        self,
        depth_msg: Image,
        depth_info_msg: CameraInfo,
        detections_msg: DetectionArray,
    ) -> None:

        new_detections_msg = DetectionArray()
        new_detections_msg.header = detections_msg.header
        new_detections_msg.detections = self.process_detections(
            depth_msg, depth_info_msg, detections_msg
        )
        self._pub.publish(new_detections_msg)

    def process_detections(
        self,
        depth_msg: Image,
        depth_info_msg: CameraInfo,
        detections_msg: DetectionArray,
    ) -> List[Detection]:

        # check if there are detections
        if not detections_msg.detections:
            return []

        transform = self.get_transform(depth_info_msg.header.frame_id)

        if transform is None:
            return []

        new_detections = []
        depth_image = self.cv_bridge.imgmsg_to_cv2(
            depth_msg, desired_encoding="passthrough"
        )

        for detection in detections_msg.detections:
            bbox3d = self.convert_bb_to_3d(depth_image, depth_info_msg, detection)

            if bbox3d is not None:
                new_detections.append(detection)

                bbox3d = Detect3DNode.transform_3d_box(bbox3d, transform[0], transform[1])
                bbox3d.frame_id = self.target_frame
                new_detections[-1].bbox3d = bbox3d

                if detection.keypoints.data:
                    keypoints3d = self.convert_keypoints_to_3d(
                        depth_image, depth_info_msg, detection
                    )
                    keypoints3d = Detect3DNode.transform_3d_keypoints(
                        keypoints3d, transform[0], transform[1]
                    )
                    keypoints3d.frame_id = self.target_frame
                    new_detections[-1].keypoints3d = keypoints3d

        return new_detections

    def convert_bb_to_3d(
        self,
        depth_image: np.ndarray,
        depth_info: CameraInfo,
        detection: Detection,
    ) -> BoundingBox3D:

        center_x = int(detection.bbox.center.position.x)
        center_y = int(detection.bbox.center.position.y)
        size_x = int(detection.bbox.size.x)
        size_y = int(detection.bbox.size.y)

        H, W = depth_image.shape[:2]
        inv_div = 1.0 / float(self.depth_image_units_divisor)

        # -- ALWAYS compute bbox ROI first (small view, not full image) ---
        u_min = max(center_x - size_x // 2, 0)
        u_max = min(center_x + size_x // 2, W)   # NOTE: end is exclusive for slicing
        v_min = max(center_y - size_y // 2, 0)
        v_max = min(center_y + size_y // 2, H)   # NOTE: end is exclusive for slicing

        if u_max <= u_min or v_max <= v_min:
            return None

        roi_raw = depth_image[v_min:v_max, u_min:u_max]  # usually uint16, view not copy
        if roi_raw.size == 0:
            return None

        # valid depth pixels (ignore zeros)
        valid = (roi_raw > 0)

        # --- If we have a segmentation polygon, mask inside the ROI (small mask) ---
        if detection.mask.data:
            poly = np.asarray(
                [[int(p.x), int(p.y)] for p in detection.mask.data],
                dtype=np.int32,
            )

            # shift polygon points into ROI coordinates
            poly[:, 0] -= u_min
            poly[:, 1] -= v_min

            # (optional safety) clip polygon into ROI bounds
            # prevents weird behavior if polygon touches outside bbox ROI
            poly[:, 0] = np.clip(poly[:, 0], 0, roi_raw.shape[1] - 1)
            poly[:, 1] = np.clip(poly[:, 1], 0, roi_raw.shape[0] - 1)

            mask_roi = np.zeros(roi_raw.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask_roi, [poly], 255)

            valid &= (mask_roi != 0)

            depths_raw = roi_raw[valid]
            if depths_raw.size == 0:
                return None

            depths_m = depths_raw.astype(np.float32) * inv_div

            # same behavior idea as before: for mask case use median depth
            bb_center_z_coord = float(np.median(depths_m))

        else:
            # no mask: use the bbox center pixel depth (same idea as your original code)
            z_center_raw = float(depth_image[center_y, center_x])
            if z_center_raw <= 0.0:
                return None

            bb_center_z_coord = z_center_raw * inv_div

            depths_raw = roi_raw[valid]
            if depths_raw.size == 0:
                return None
            depths_m = depths_raw.astype(np.float32) * inv_div

        # --- filter around the chosen center depth ---
        keep = np.abs(depths_m - bb_center_z_coord) <= float(self.maximum_detection_threshold)
        if not np.any(keep):
            return None

        depths_m = depths_m[keep]
        if depths_m.size == 0:
            return None

        z_min = float(depths_m.min())
        z_max = float(depths_m.max())
        z = 0.5 * (z_min + z_max)

        if z <= 0.0:
            return None

        # --- project bbox center + bbox size into 3D (same math as before) ---
        k = depth_info.k
        px, py, fx, fy = k[2], k[5], k[0], k[4]

        x = z * (center_x - px) / fx
        y = z * (center_y - py) / fy
        w = z * (size_x / fx)
        h = z * (size_y / fy)

        msg = BoundingBox3D()
        msg.center.position.x = x
        msg.center.position.y = y
        msg.center.position.z = z
        msg.size.x = w
        msg.size.y = h
        msg.size.z = float(z_max - z_min)

        # --- orientation (unchanged from your current code) ---
        pts = self._sample_points_3d(depth_image, depth_info, detection, stride=4, max_points=4000)

        msg.center.orientation.x = 0.0
        msg.center.orientation.y = 0.0
        msg.center.orientation.z = 0.0
        msg.center.orientation.w = 1.0
        
        if pts is not None and pts.shape[0] >= self.min_seg_points_for_orientation:
            frame = Detect3DNode._plane_frame_from_pts_pca(pts)
            if frame is not None:
                z_axis, x_axis, y_axis = frame  # z_axis is the normal

                R = np.column_stack([x_axis, y_axis, z_axis])
                q_xyzw = Rot.from_matrix(R).as_quat()

                msg.center.orientation.x = float(q_xyzw[0])
                msg.center.orientation.y = float(q_xyzw[1])
                msg.center.orientation.z = float(q_xyzw[2])
                msg.center.orientation.w = float(q_xyzw[3])

        return msg

    def convert_keypoints_to_3d(
        self,
        depth_image: np.ndarray,
        depth_info: CameraInfo,
        detection: Detection,
    ) -> KeyPoint3DArray:

        # build an array of 2d keypoints
        keypoints_2d = np.array(
            [[p.point.x, p.point.y] for p in detection.keypoints.data], dtype=np.int16
        )
        u = np.array(keypoints_2d[:, 1]).clip(0, depth_info.height - 1)
        v = np.array(keypoints_2d[:, 0]).clip(0, depth_info.width - 1)

        # sample depth image and project to 3D
        z = depth_image[u, v]
        k = depth_info.k
        px, py, fx, fy = k[2], k[5], k[0], k[4]
        x = z * (v - px) / fx
        y = z * (u - py) / fy
        points_3d = (
            np.dstack([x, y, z]).reshape(-1, 3) / self.depth_image_units_divisor
        )  # convert to meters

        # generate message
        msg_array = KeyPoint3DArray()
        for p, d in zip(points_3d, detection.keypoints.data):
            if not np.isnan(p).any():
                msg = KeyPoint3D()
                msg.point.x = p[0]
                msg.point.y = p[1]
                msg.point.z = p[2]
                msg.id = d.id
                msg.score = d.score
                msg_array.data.append(msg)

        return msg_array

    def get_transform(self, frame_id: str) -> Tuple[np.ndarray]:
        # transform position from image frame to target_frame
        rotation = None
        translation = None

        try:
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                self.target_frame, frame_id, rclpy.time.Time()
            )

            translation = np.array(
                [
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z,
                ]
            )

            rotation = np.array(
                [
                    transform.transform.rotation.w,
                    transform.transform.rotation.x,
                    transform.transform.rotation.y,
                    transform.transform.rotation.z,
                ]
            )

            return translation, rotation

        except TransformException as ex:
            self.get_logger().error(f"Could not transform: {ex}")
            return None
    @staticmethod
    def _quat_is_identity(q_wxyz, tol=1e-3) -> bool:
        q = np.asarray(q_wxyz, dtype=np.float64)
        q = Detect3DNode._quat_normalize(q)
        # identity is [±1,0,0,0]
        return (abs(q[1]) < tol and abs(q[2]) < tol and abs(q[3]) < tol and abs(abs(q[0]) - 1.0) < tol)

    @staticmethod
    def transform_3d_box(
        bbox: BoundingBox3D,
        translation: np.ndarray,
        rotation: np.ndarray,
    ) -> BoundingBox3D:
        """
        Transform bbox center pose from source frame to target_frame.

        - translation: (3,) in target_frame
        - rotation: quaternion [w, x, y, z] that rotates vectors from source->target
        """

        # ---------------------------
        # Position: p' = R * p + t
        # ---------------------------
        p_src = np.array(
            [
                bbox.center.position.x,
                bbox.center.position.y,
                bbox.center.position.z,
            ],
            dtype=np.float64,
        )

        p_tgt = Detect3DNode.qv_mult(rotation, p_src) + np.asarray(translation, dtype=np.float64)

        bbox.center.position.x = float(p_tgt[0])
        bbox.center.position.y = float(p_tgt[1])
        bbox.center.position.z = float(p_tgt[2])

        # ---------------------------
        # Orientation
        #
        # We treat identity bbox orientation as a SENTINEL:
        # "orientation was not reliably computed" (e.g., too few seg points).
        #
        # In that case, KEEP identity in the target frame (or swap in your
        # own predefined quaternion if you prefer).
        # ---------------------------

        # bbox quaternion from message (wxyz)
        q_bbox = np.array(
            [
                bbox.center.orientation.w,
                bbox.center.orientation.x,
                bbox.center.orientation.y,
                bbox.center.orientation.z,
            ],
            dtype=np.float64,
        )

        # Handle "unset all zeros" safely
        if np.linalg.norm(q_bbox) < 1e-12:
            q_bbox[:] = (1.0, 0.0, 0.0, 0.0)

        # If YOLO3D couldn't compute orientation and left identity,
        # don't propagate TF rotation into it.
        if Detect3DNode._quat_is_identity(q_bbox, tol=1e-3):
            # option A: keep identity
            bbox.center.orientation.x = 0.0
            bbox.center.orientation.y = 0.0
            bbox.center.orientation.z = 0.0
            bbox.center.orientation.w = 1.0

            # option B: use a predefined target-frame orientation instead
            # (uncomment if you want)
            # bbox.center.orientation.x = PREDEF_X
            # bbox.center.orientation.y = PREDEF_Y
            # bbox.center.orientation.z = PREDEF_Z
            # bbox.center.orientation.w = PREDEF_W
        else:
            # q_target = q_tf ⊗ q_bbox
            q_new = Detect3DNode._quat_multiply(rotation, q_bbox)
            q_new = Detect3DNode._quat_normalize(q_new)
            bbox.center.orientation.w = float(q_new[0])
            bbox.center.orientation.x = float(q_new[1])
            bbox.center.orientation.y = float(q_new[2])
            bbox.center.orientation.z = float(q_new[3])

        return bbox

    @staticmethod
    def transform_3d_keypoints(
        keypoints: KeyPoint3DArray,
        translation: np.ndarray,
        rotation: np.ndarray,
    ) -> KeyPoint3DArray:

        for point in keypoints.data:
            position = (
                Detect3DNode.qv_mult(
                    rotation, np.array([point.point.x, point.point.y, point.point.z])
                )
                + translation
            )

            point.point.x = position[0]
            point.point.y = position[1]
            point.point.z = position[2]

        return keypoints

    @staticmethod
    def qv_mult(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        q = np.asarray(q)
        v = np.asarray(v)
        qvec = q[1:]
        uv = np.cross(qvec, v)
        uuv = np.cross(qvec, uv)
        return v + 2.0 * (uv * q[0] + uuv)
    
    @staticmethod
    def _quat_normalize(q_wxyz):
        q = np.array(q_wxyz, dtype=np.float64)
        n = np.linalg.norm(q)
        if n < 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return q / n

    @staticmethod
    def _quat_multiply(q1_wxyz, q2_wxyz):
        w1, x1, y1, z1 = q1_wxyz
        w2, x2, y2, z2 = q2_wxyz
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dtype=np.float64)

    @staticmethod
    def _plane_frame_from_pts_pca(
        pts: np.ndarray,
        x_ref: np.ndarray = np.array([1.0, 0.0, 0.0], dtype=np.float64),
    ):
        """
        Returns:
          n: unit normal (3,)
          x: unit in-plane major axis (3,)
          y: unit in-plane axis completing right-handed frame (3,)
        """

        c = np.mean(pts, axis=0)
        Q = pts - c

        cov = (Q.T @ Q) / len(pts)  # safe since caller ensures N>=50
        w, V = np.linalg.eigh(cov)  # w ascending, V columns are eigenvectors

        # Normal = smallest variance direction
        n = V[:, 0]

        # Major in-plane axis = largest variance direction
        x = V[:, 2]

        # Make normal direction consistent (camera at origin in camera frame)
        if np.dot(n, c) > 0:
            n = -n

        # Build a right-handed orthonormal basis
        # (use cross products so we stay consistent even after flips)
        y = np.cross(n, x)
        yn = np.linalg.norm(y)
        if yn < 1e-12:
            return None  # degenerate
        y = y / yn

        x = np.cross(y, n)
        x = x / (np.linalg.norm(x) + 1e-12)

        # Make x direction deterministic to reduce 180° yaw flips
        if np.dot(x, x_ref) < 0:
            x = -x
            y = -y  # keep frame right-handed (n stays the same)

        return n, x, y

    def _sample_points_3d(self, depth_image, depth_info, detection, stride=4, max_points=4000):
        k = depth_info.k
        cx, cy, fx, fy = k[2], k[5], k[0], k[4]
        inv_div = 1.0 / float(self.depth_image_units_divisor)

        h_img, w_img = depth_image.shape[:2]

        # --- define ROI from bbox (always) ---
        center_x = int(detection.bbox.center.position.x)
        center_y = int(detection.bbox.center.position.y)
        size_x = int(detection.bbox.size.x)
        size_y = int(detection.bbox.size.y)

        u_min = max(center_x - size_x // 2, 0)
        u_max = min(center_x + size_x // 2, w_img)
        v_min = max(center_y - size_y // 2, 0)
        v_max = min(center_y + size_y // 2, h_img)

        if u_max <= u_min or v_max <= v_min:
            return None

        roi = depth_image[v_min:v_max, u_min:u_max]

        # --- pixels to sample inside ROI ---
        if detection.mask.data:
            # build a SMALL mask just for the ROI
            poly = np.array([[int(p.x), int(p.y)] for p in detection.mask.data], dtype=np.int32)

            # shift polygon into ROI coordinates
            poly[:, 0] -= u_min
            poly[:, 1] -= v_min

            mask = np.zeros(roi.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [poly], 255)

            # stride directly on mask to reduce points early
            m = mask[::stride, ::stride]
            ys, xs = np.where(m > 0)

            # scale coords back up to ROI pixel coords
            ys = ys * stride
            xs = xs * stride
        else:
            ys = np.arange(0, roi.shape[0], stride, dtype=np.int32)
            xs = np.arange(0, roi.shape[1], stride, dtype=np.int32)
            ys, xs = np.meshgrid(ys, xs, indexing="ij")
            ys = ys.ravel()
            xs = xs.ravel()

        if ys.size == 0:
            return None

        # sample depth (still in ROI coords)
        z_raw = roi[ys, xs].astype(np.float32)
        z = z_raw * inv_div

        # valid depth
        valid = (z > 0.0)
        # if your depth image can contain NaN, keep np.isfinite too:
        # valid &= np.isfinite(z)

        if not np.any(valid):
            return None

        z = z[valid]
        if z.size < 50:
            return None
        ys = ys[valid]; xs = xs[valid]; 
    
        # outlier rejection around median depth
        z_med = np.median(z)
        keep = np.abs(z - z_med) <= float(self.maximum_detection_threshold)
        if not np.any(keep):
            return None

        z = z[keep]
        if z.size < 50:
            return None
        ys = ys[keep]; xs = xs[keep]; 
    
        # cap number of points BEFORE computing X/Y
        if z.size > max_points:
            idx = np.random.choice(z.size, size=max_points, replace=False)
            ys = ys[idx]; xs = xs[idx]; z = z[idx]

        # convert ROI pixels to full-image pixels
        xs_img = xs + u_min
        ys_img = ys + v_min

        # backproject
        X = z * (xs_img - cx) / fx
        Y = z * (ys_img - cy) / fy
        return np.column_stack((X, Y, z))

def main():
    rclpy.init()
    node = Detect3DNode()
    node.trigger_configure()
    node.trigger_activate()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
