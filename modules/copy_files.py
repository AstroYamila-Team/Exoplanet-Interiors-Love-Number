import os, shutil

def copyfiles(src, dst):
    # copying the symbolic link still doesn't really work without copying the directory
    files = os.listdir(src)
    for i in range(len(files)):
        if os.path.isdir(src+files[i]):
            if os.path.islink(src+files[i]):
                # shutil.copy(src+files[i], planet_path+'/', follow_symlinks=True)
                linkto = os.readlink(src+files[i])
                os.symlink('/home/s2034174/python_test_env/CEPAM/data', dst+'data')
                # print('this happens')
            else:
                # print(files[i])
                shutil.copytree(src+files[i], dst+files[i])
        else:
            shutil.copy(src+files[i], dst, follow_symlinks=False)