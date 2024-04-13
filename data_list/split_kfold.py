import os
import shutil
root_dir = "data_list/pacs/pacs_data/pacs_data"
instr_dir = "data_list/PACS_list"
modes = ["train", "crossval"]

for mode in modes:
    for root, _, fnames in os.walk(os.path.join(instr_dir, mode)):
        for fname in fnames:
            with open(os.path.join(root, fname)) as f: 
                lines = f.readlines()
                for line in lines:
                    instr, _ = line.split()
                    src = os.path.join(root_dir, instr)
                    dst = os.path.join(root_dir + "_" + mode, instr)
                    dstpath, _=os.path.split(dst)
                    if not os.path.exists(dstpath):
                        os.makedirs(dstpath)
                    shutil.copy(src, dst) 
                    print("Copying from {} to {}".format(src, dst))


    



