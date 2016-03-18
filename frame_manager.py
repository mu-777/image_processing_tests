#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np


class FaceManager(object):
    def __init__(self, face):
        (x, y, w, h) = face
        self.center_x = x + int(w / 2)
        self.center_y = y + int(h / 2)
        self.face_size = np.sqrt(w ** 2 + h ** 2)
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def get_same_face(self, face_mgrs, allowed_rate=0.05):
        def calc_diff(face1, face2):
            return np.sqrt((face1.center_x - face2.center_x) ** 2 + (face1.center_y - face2.center_y) ** 2)

        diffs = [calc_diff(self, face) for face in face_mgrs]
        min_diff = min(diffs)
        min_idx = diffs.index(min_diff)
        if min_diff < self.face_size * allowed_rate:
            return face_mgrs[min_idx]

    def average(self, face_mgr):
        x = (self.x + face_mgr.x) / 2.0
        y = (self.y + face_mgr.y) / 2.0
        w = (self.w + face_mgr.w) / 2.0
        h = (self.h + face_mgr.h) / 2.0
        return FaceManager((x, y, w, h))


class FacesManager(object):
    def __init__(self):
        self._front_frame_faces = []
        self._center_frame_faces = []
        self._rear_frame_faces = []

    def append(self, faces):
        if len(self._front_frame_faces) == 0:
            self._front_frame_faces = self._set_faces(faces)
            self._center_frame_faces = self._set_faces(faces)
            self._rear_frame_faces = self._set_faces(faces)

        self._front_frame_faces = self._center_frame_faces
        self._center_frame_faces = self._rear_frame_faces
        self._rear_frame_faces = self._set_faces(faces)
        return self

    def get_faces(self):
        continuous_faces = self._get_continuous_faces()
        return self._added_faces(continuous_faces)

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
                self._center_frame_faces.append(face)
        return self._center_frame_faces
