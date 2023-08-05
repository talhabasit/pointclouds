import json
import open3d as o3d


"""This script is used to read the calibration file and return the intrinsics matrix as output
Note: The Intrinsics have to be saved using the realsense-viewer in json format as these are sensor specific
"""

def get_intrinsics_from_json(number=1):
    """This function returns open3d intrinsics matrix as output corresponding to the selected number
    #The numbers are as follow: 0)1920x1080 1)1280x720 2)640x480 3)848x480 4)640x360"""

    with open("./Calib_file.json", "r") as f:
        data = json.load(f)

    w = int(data["rectified.{}.width".format(number)])
    h = int(data["rectified.{}.height".format(number)])

    fx = float(data["rectified.{}.fx".format(number)])
    fy = float(data["rectified.{}.fy".format(number)])
    ppx = float(data["rectified.{}.ppx".format(number)])
    ppy = float(data["rectified.{}.ppy".format(number)])

    return (
        o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, ppx, ppy),
        fx,
        fy,
        ppx,
        ppy,
        w,
        h,
    )


if __name__ == "__main__":
    x = get_intrinsics_from_json(1)
    print(x)
    v = 0
