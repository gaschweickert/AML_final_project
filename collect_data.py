
import os
import pandas as pd
import numpy as np


def collect_data_from_file(directory, filename):
    print(os.path.join(directory, filename))
    data_df = pd.DataFrame({"image_loc": [],"image_id": [], "bug_type": [], "count":[]})
    for number, line in enumerate(open(os.path.join(directory, filename))):
        if number == 0:
            continue
        try:
            im_loc = "./bug_images/"+line.strip().split(",")[0]
            # im_loc = os.path.join(__location__, "bug_images/"+line.strip().split(",")[0])
            data_df.loc[len(data_df.index)] = [im_loc, line.strip().split(",")[0], line.strip().split(",")[1], line.strip().split(",")[2]]
        except Exception as e:
            continue
    

    data_abw  = data_df[(data_df["bug_type"]=='abw')]
    data_abw = pd.concat([data_abw, data_df[~data_df["image_id"].isin(data_abw["image_id"])]])
    data_abw.loc[data_abw["bug_type"]=='pbw', "count"] = 0
    data_abw.drop('bug_type', inplace = True, axis = 1)

    data_pbw  = data_df[(data_df["bug_type"]=='pbw')]
    data_pbw = pd.concat([data_pbw, data_df[~data_df["image_id"].isin(data_pbw["image_id"])]])
    data_pbw.loc[data_pbw["bug_type"]=='abw', "count"] = 0
    data_pbw.drop('bug_type', inplace = True, axis = 1) 

    grouped_data_df = pd.DataFrame({"image_loc": [],"image_id": [], "counts":[]})
    for id in data_df.image_id.unique():
        abw_count = data_abw[(data_abw.image_id == id)]["count"].to_numpy()
        pbw_count = data_pbw[(data_pbw.image_id == id)]["count"].to_numpy()
        grouped_data_df.loc[len(grouped_data_df.index)] = [os.path.join(directory, "bug_images/"+id), id, np.asarray([abw_count[0], pbw_count[0]]).astype('float64')]
    return grouped_data_df, data_df, data_abw, data_pbw

if __name__ == "__main__":
    directory = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    grouped_data_df, data_df, data_abw, data_pbw = collect_data_from_file(directory, 'Train.csv')
    grouped_data_df.to_csv(os.path.join(directory, 'dataframes/grouped_data_df.csv'))
    data_df.to_csv(os.path.join(directory, 'dataframes/train_df.csv'))
    data_abw.to_csv(os.path.join(directory, 'dataframes/data_abw.csv'))
    data_pbw.to_csv(os.path.join(directory, 'dataframes/data_pbw.csv'))