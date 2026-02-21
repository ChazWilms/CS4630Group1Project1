import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser(description="Philly 311 dataset cleaning program")
parser.add_argument(
        '-d', '--dir', 
        type=str, 
        required=True,
        help="Target directory path (default: H:/pythonClass2/) [MUST include the final '/']"
    )
    
parser.add_argument(
    '-f', '--filename', 
    type=str, 
    required=True,
    help="Target filename (default: public_cases_fc.csv)"
)


args = parser.parse_args()
dir = args.dir
filename = args.filename
    
print(f"{dir} + {filename}")
exit(4)


#information of columns that are in the dataset and what should be removed
headers = ['objectid','service_request_id','subject','status','status_notes','service_name','service_code','agency_responsible','service_notice','requested_datetime','updated_datetime','expected_datetime','closed_datetime','address','zipcode','media_url','lat','lon']
cols_to_remove = ['media_url', 'expected_datetime', 'updated_datetime', 'closed_datetime', 'status', 'service_notice']

#remove the previously defined columns
def removeColumns(dir_):
    for filename_ in os.listdir(dir_):
        if filename_.endswith('.csv'):
            file_path = os.path.join(dir_, filename_)

            df = pd.read_csv(file_path, skiprows=1, header=None)

            if len(df.columns) == len(headers):
                df.columns = headers

                df = df.drop(columns=cols_to_remove)

                df.to_csv(file_path, index=False)
                print(f"Removed columns from {filename_}")
            else:
                print(f"Error on {filename_}")



def cleanDataFile(dir, filename):
    df = pd.read_csv(dir+filename)

    beforeCleanLen = len(df)

    clean = df.dropna(subset=['lat', 'lon'], how='any')
    clean = clean.dropna(subset=['service_name'])
    clean = clean.dropna(subset=['subject', 'status_notes'], how='all')


    afterCleanLen = len(clean)

    print(f"Dropped {beforeCleanLen - afterCleanLen} rows missing  data.")
    print(f"Remaining rows: {afterCleanLen}")

    clean.to_csv(dir+filename, index=False)



def deduplicateDataFile(dir, filename):
    df = pd.read_csv(dir+filename)

    beforeDedupLen = len(df)

    deDuped = df.drop_duplicates(subset=['service_request_id'])

    print(f"Removed {beforeDedupLen - len(deDuped)} identical 'service_request_id'")

    deDuped['requested_service_day'] = pd.to_datetime(deDuped['requested_datetime']).dt.date

    #round to about 11 meters to get complaints about the same events
    deDuped['lat_round'] = deDuped['lat'].round(4)
    deDuped['lon_round'] = deDuped['lon'].round(4)

    #drop other records other than the first mention
    duplicateDataSubset = ['service_name', 'lat_round', 'lon_round', 'requested_service_day']
    deDuped = deDuped.drop_duplicates(subset=duplicateDataSubset, keep='first')

    #remove the extra added columns
    deDuped = deDuped.drop(columns=['requested_service_day', 'lat_round', 'lon_round'])

    afterDedupLen = len(deDuped)

    print(f"Removed {beforeDedupLen - afterDedupLen} duplicate records.")
    print(f"Remaining distinct events: {afterDedupLen}")

    deDuped.to_csv(dir+filename, index=False)


def normalizeComplaintType(dir, filename):
    df = pd.read_csv(dir+filename)

    df_normalize = df.copy()

    #treat as str, make everything lowercase, and remove trailing/leading whitespace
    df_normalize['service_name'] = df_normalize['service_name'].astype(str)
    df_normalize['service_name'] = df_normalize['service_name'].str.lower()
    df_normalize['service_name'] = df_normalize['service_name'].str.strip()
    #use regex to eliminate any unnecessary spacing in the service name
    df_normalize['service_name'] = df_normalize['service_name'].str.replace(r'\s+', ' ', regex=True)

    #treat as str, make everything lowercase, and remove trailing/leading whitespace
    df_normalize['subject'] = df_normalize['subject'].astype('str')
    df_normalize['subject'] = df_normalize['subject'].str.lower()
    df_normalize['subject'] = df_normalize['subject'].str.strip()
    #use regex to eliminate any unnecessary spacing in the service name
    df_normalize['subject'] = df_normalize['subject'].str.replace(r'\s+', ' ', regex=True)


    df_normalize.to_csv(dir+filename, index=False)


def validateZipCodes(dir, filename):
    df = pd.read_csv(dir + filename)

    df_zip = df.copy()

    df_zip['zipcode'] = df_zip['zipcode'].astype(str)

    #remove from it previously being treated as a float
    df_zip['zipcode'] = df_zip['zipcode'].str.replace(r'\.0$', '', regex=True)

    #get first 5 digits to remove extentions
    df_zip['zipcode'] = df_zip['zipcode'].str.extract(r'(\d{5})')[0]

    invalidZips = df_zip['zipcode'].isna().sum()

    print(f"Cleaned zipcodes. {invalidZips} records have invalid or missing zipcodes (set to NaN, BUT not removed).")

    df_zip.to_csv(dir+filename, index=False)

removeColumns(dir)
cleanDataFile(dir, filename)
deduplicateDataFile(dir, filename)
normalizeComplaintType(dir, filename)
validateZipCodes(dir, filename)