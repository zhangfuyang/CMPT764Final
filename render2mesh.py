import numpy as np
import os
from numpy import linalg as LA
import matplotlib.pyplot as plt

datapath = '../Chair/models'
def renderBoxes2mesh(boxes, obj_names):
    obj_name_set = set(obj_names)
    obj_dict = {}
    for idx, name in enumerate(obj_name_set):
        vertices = []
        faces = []
        with open(os.path.join(datapath, name), 'r') as f:
            lines = f.readlines()
        for line in lines:
            if line[0] != 'v' and line[0] != 'f':
                continue
            line = line.strip('\n')
            items = line.split(' ')
            if items[0] == 'v':
                vertices.append((float(items[1]), float(items[2]), float(items[3])))
            if items[0] == 'f':
                faces.append((int(items[1]), int(items[2]), int(items[3])))

        vertices = np.array(vertices)
        obj_dict[name] = {'vertices': vertices, 'faces': faces, 'id': idx}


    results = []
    for box_i in range(boxes.shape[0]):
        box = boxes[box_i]
        obj = obj_dict[obj_names[box_i]]
        vertices = obj['vertices']
        faces = obj['faces']
        center = box[0:3]
        lengths = box[3:6] * 1.1
        dir_1 = box[6:9]
        dir_2 = box[9:12]
        dir_1 = dir_1/LA.norm(dir_1)
        dir_2 = dir_2/LA.norm(dir_2)
        dir_3 = np.cross(dir_1, dir_2)

        dist_v = vertices - center
        dist_1 = np.abs(np.dot(dist_v, dir_1))
        dist_2 = np.abs(np.dot(dist_v, dir_2))
        dist_3 = np.abs(np.dot(dist_v, dir_3))
        clean_flag = np.logical_and(dist_1 <= lengths[0] / 2, dist_2 <= lengths[1] / 2)
        clean_flag = np.logical_and(clean_flag, dist_3 <= lengths[2] / 2)

        new_id = [0 for _ in range(vertices.shape[0])]
        count = 0
        new_v = []
        new_f = []
        for i in range(vertices.shape[0]):
            if clean_flag[i]:
                count += 1
                new_id[i] = count
                new_v.append(vertices[i])
        for i in range(len(faces)):
            a = faces[i][0]
            b = faces[i][1]
            c = faces[i][2]
            if clean_flag[a-1] and clean_flag[b-1] and clean_flag[c-1]:
                new_f.append([new_id[a-1], new_id[b-1], new_id[c-1]])
        results.append((new_v, new_f))

    return results

def saveOBJ(obj_names, outfilename, results):
    cmap = plt.get_cmap('jet_r')
    obj_name_set = set(obj_names)
    obj_dict = {}
    for idx, name in enumerate(obj_name_set):
        obj_dict[name] = idx
    f = open(outfilename, 'w')
    offset = 0
    for box_i in range(len(results)):
        color = cmap(float(obj_dict[obj_names[box_i]]) / len(obj_name_set))[:-1]
        vertices = results[box_i][0]
        faces = results[box_i][1]
        for i in range(len(vertices)):
            f.write('v ' + str(vertices[i][0]) + ' ' + str(vertices[i][1]) + ' ' + str(vertices[i][2]) +
                    ' ' + str(color[0]) + ' ' + str(color[1]) + ' ' + str(color[2]) + '\n')
        for i in range(len(faces)):
            f.write('f ' + str(faces[i][0]+offset) + ' ' + str(faces[i][1]+offset) + ' ' + str(faces[i][2]+offset) + '\n')
        offset += len(vertices)
    f.close()

def directRender(boxes, obj_names, outfilename):
    results = renderBoxes2mesh(boxes, obj_names)
    saveOBJ(obj_names, outfilename, results)


def alignBoxAndRender(gtBoxes, predBoxes, obj_names, outfilename):
    results = renderBoxes2mesh(gtBoxes, obj_names)
    for i in range(len(results)):
        gtbox = gtBoxes[i]
        gtCenter = gtbox[0:3][np.newaxis, ...].T
        gtlengths = gtbox[3:6]
        gtdir_1 = gtbox[6:9]
        gtdir_2 = gtbox[9:12]
        gtdir_1 = gtdir_1/LA.norm(gtdir_1)
        gtdir_2 = gtdir_2/LA.norm(gtdir_2)
        gtdir_3 = np.cross(gtdir_1, gtdir_2)

        predbox = predBoxes[i]
        predCenter = predbox[0:3][np.newaxis, ...].T
        predlengths = predbox[3:6]
        preddir_1 = predbox[6:9]
        preddir_2 = predbox[9:12]
        preddir_1 = preddir_1/LA.norm(preddir_1)
        preddir_2 = preddir_2/LA.norm(preddir_2)
        preddir_3 = np.cross(preddir_1, preddir_2)

        scale = predlengths / gtlengths
        scale = np.array([[scale[0], 0, 0], [0, scale[1], 0], [0, 0, scale[2]]])
        x = np.array(results[i][0]).T
        A = np.array([gtdir_1, gtdir_2, gtdir_3])
        B = np.array([preddir_1, preddir_2, preddir_3])
        B = B.T
        y = scale.dot(B).dot(A).dot(x-gtCenter)+predCenter
        x = y.T
        for t in range(len(results[i][0])):
            results[i][0][t] = x[t]
    saveOBJ(obj_names, outfilename, results)




