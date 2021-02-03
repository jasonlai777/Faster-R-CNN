import os.path as osp
import sys

def add_path(path,n):
    if path not in sys.path:
        sys.path.insert(n, path)

this_dir = osp.dirname(__file__)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, 'lib')
add_path(lib_path,1)

sr_path = osp.join(this_dir, 'srgan')
add_path(sr_path,2)

coco_path = osp.join(this_dir, 'data', 'coco', 'PythonAPI')
add_path(coco_path,0)
