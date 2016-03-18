#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np


class FaceManager(object):
    def __init__(self, face):
        (x, y, w, h) = face
        self.center_x = x + int(w / 2)
        self.center_y = y + int(h / 2)
        self.size = np.sqrt(w ** 2 + h ** 2)
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def get_same_face(self, face_mgrs, allowed_rate=0.05):
        def calc_diff(face1, face2):
            return np.sqrt((face1.center_x - face2.center_x) ** 2 + (face1.center_y - face2.center_y) ** 2)

        if len(face_mgrs) == 0:
            return None

        diffs = [calc_diff(self, face) for face in face_mgrs]
        min_diff = min(diffs)
        min_idx = diffs.index(min_diff)
        min_face_mgr = face_mgrs[min_idx]

        is_center_close = min_diff < self.size * allowed_rate
        is_size_close = abs(min_face_mgr.size - self.size) < self.size * allowed_rate

        if is_center_close and is_size_close:
            return min_face_mgr
        else:
            return None

    def average(self, face_mgr):
        x = int((self.x + face_mgr.x) / 2.0)
        y = int((self.y + face_mgr.y) / 2.0)
        w = int((self.w + face_mgr.w) / 2.0)
        h = int((self.h + face_mgr.h) / 2.0)
        return FaceManager((x, y, w, h))

    @property
    def face(self):
        return int(self.x), int(self.y), int(self.w), int(self.h)


class FacesManager(object):
    def __init__(self):
        self._front_frame = None
        self._center_frame = None
        self._rear_frame = None

        self._front_frame_faces = []
        self._center_frame_faces = []
        self._rear_frame_faces = []

    def initialize(self, frame, faces):
        self._front_frame = frame
        self._center_frame = frame
        self._rear_frame = frame
        self._front_frame_faces = self._set_faces(faces)
        self._center_frame_faces = self._set_faces(faces)
        self._rear_frame_faces = self._set_faces(faces)

    def append(self, frame, faces):
        if self._front_frame is None:
            self.initialize(frame, faces)
            return self

        self._front_frame_faces = self._center_frame_faces
        self._center_frame_faces = self._rear_frame_faces
        self._rear_frame_faces = self._set_faces(faces)

        self._front_frame = self._center_frame
        self._center_frame = self._rear_frame
        self._rear_frame = frame
        return self

    def get(self):
        continuous_faces = self._get_continuous_faces()
        updated_faces = self._added_faces(continuous_faces)
        return self._center_frame, [updated_face.face for updated_face in updated_faces]

    def _set_faces(self, faces):
        ret_list = []
        for face in faces:
            ret_list.append(FaceManager(face))
        return ret_list

    def _get_continuous_faces(self):
        continuous_faces = []
        for face in self._front_frame_faces:
            same_face = face.get_same_face(self._rear_frame_faces)
            if same_face is not None:
                continuous_faces.append(face.average(same_face))
        return continuous_faces

    def _added_faces(self, faces):
        for face in faces:
            same_face = face.get_same_face(self._center_frame_faces)
            if same_face is None:
                print("interpolate!")
                self._center_frame_faces.append(face)
        return self._center_frame_faces
