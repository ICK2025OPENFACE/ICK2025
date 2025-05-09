from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import mediapipe as mp
import numpy as np
import socket


def draw_landmarks_on_image(rgb_image, detection_result):
    """
    Draws facial landmarks on an RGB image based on the detection results.
    This function takes an input RGB image and a detection result containing
    facial landmarks, and it visualizes the landmarks on the image. The landmarks
    are drawn using MediaPipe's drawing utilities, including tessellation, contours,
    and iris connections.
    Args:
        rgb_image (numpy.ndarray): The input RGB image on which the landmarks will be drawn.
        detection_result: The detection result containing facial landmarks. It is expected
                          to have a `face_landmarks` attribute, which is a list of landmarks
                          for each detected face.
    Returns:
        numpy.ndarray: The annotated image with facial landmarks drawn on it.
    """

    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in face_landmarks
            ]
        )

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style(),
        )
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style(),
        )

    return annotated_image


def send_msg_via_udp(msg: str, udp_socket:socket.socket, server_ip: str, server_port: str) -> None:
    """
    Sends a message via UDP to a specified server.
    Args:
        msg (str): The message to be sent. It will be converted to a string if not already.
        udp_socket (socket.socket): The UDP socket object used to send the message.
        server_ip (str): The IP address of the server to send the message to.
        server_port (str): The port number of the server to send the message to.
    Returns:
        None
    Raises:
        Exception: Catches and prints any unhandled exceptions that occur during the sending process.
    """
    
    try:
        if not msg is None:
            msg = str(msg)
            udp_socket.sendto(msg.encode("ascii"), (server_ip, server_port))
    except Exception as e:
        print(f"Unhandled exception in supportfunctions.send_msg_via_udp function: {e}")
