import csv

import app
import pathfinder

def write(id2prediction, submission_path):
    """
    :param id2prediction: dict of {img_id: prediction vector}
    :param submission_path:
    """
    headers = app.get_headers()
    id2tags = {}
    for iid in id2prediction.keys():
        row = []
        for idx, class_value in enumerate(id2prediction[iid]):
            if class_value>0.5:
                row.append(headers[idx])
        tags = ' '.join(row)
        id2tags[iid] = tags


    f_ss = open(pathfinder.SAMPLE_SUBMISSION_PATH)
    ss = csv.reader(f_ss)

    f = open(submission_path, 'w+')
    fo = csv.writer(f, lineterminator='\n')
    fo.writerow(['image_name', 'tags'])

    for irow, row in enumerate(ss):
        if irow == 0:
            continue
        iid = row[0]
        fo.writerow([iid, id2tags[iid]])
    
    f_ss.close()
    f.close()



