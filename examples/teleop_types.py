import ctypes


class Twist(ctypes.Structure):
    _fields_ = [
        ("vx", ctypes.c_float),
        ("vy", ctypes.c_float),
        ("vz", ctypes.c_float),
        ("wx", ctypes.c_float),
        ("wy", ctypes.c_float),
        ("wz", ctypes.c_float),
    ]

    @staticmethod
    def type_name() -> str:
        return "Twist"


class Pose(ctypes.Structure):
    _fields_ = [
        ("qw", ctypes.c_float),
        ("qx", ctypes.c_float),
        ("qy", ctypes.c_float),
        ("qz", ctypes.c_float),
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
    ]

    @staticmethod
    def type_name() -> str:
        return "Pose"
