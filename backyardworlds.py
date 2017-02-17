import sys
import pandas as pd
import json
from astropy.io import ascii

def generate_CSV(classfile_in, markfile_out):
    """
    Generates a readable CSV file from exported Zooniverse data
    
    Parameters
    ----------
    classfile_in: str
        The file to convert
    markfile_out: str
        The path and filename of the output file
    
    # -------------------------------------------------------------
    # Panoptes Marking Export Script
    #
    # This script extracts individual markings from Zooniverse
    # Panoptes classification data export CSV.  This script is 
    # configured to export circular marker info for classifications
    # collected only for the latest workflow version.
    #
    # Customizations are set for use with the following project:
    # planet-9-rogue-worlds
    #
    # Column names, annotation info, and marking task ID may need
    # be altered for this script to work for data exports from
    # other projects.
    #
    # Written by: Cliff Johnson (lcj@ucsd.edu)
    # Last Edited: 10 January 2017
    # Based on scripts by Brooke Simmons 
    # -------------------------------------------------------------
    
    """
    # Read in classification CSV and expand JSON fields
    classifications = pd.read_csv(classfile_in)
    classifications['metadata_json'] = [json.loads(q) for q in classifications.metadata]
    classifications['annotations_json'] = [json.loads(q) for q in classifications.annotations]
    classifications['subject_data_json'] = [json.loads(q) for q in classifications.subject_data]

    # Calculate number of markings per classification
    # Note: index of annotations_json ("q" here) corresponds to task number (i.e., 0)
    classifications['n_markings'] = [ len(q[0]['value']) for q in classifications.annotations_json ]

    ### Classification Selection / CURRENT SETTING: most recent workflow version
    # OPTION 1: Select only classifications from most recent workflow version
    iclass = classifications[classifications.workflow_version == classifications['workflow_version'].max()]
    # OPTION 2: Select most/all valid classifications using workflow_id and workflow_version
    #iclass = classifications[(classifications['workflow_id'] == 1687) & (classifications['workflow_version'] > 40)]

    # Output markings from classifications in iclass to new list of dictionaries (prep for pandas dataframe)
    # Applicable for workflows with marking task as first task, and outputs data for circular markers (x,y,r)
    clist=[]
    for index, c in iclass.iterrows():
        if c['n_markings'] > 0:
            # Note: index of annotations_json corresponds to task number (i.e., 0)
            for q in c.annotations_json[0]['value']:
            
                # OPTIONAL EXPANSION: could use if statement here to split marker types
                clist.append({'classification_id':c.classification_id, 'user_name':c.user_name, 'user_id':c.user_id,
                              'created_at':c.created_at, 'subject_ids':c.subject_ids, 'tool':q['tool'], 
                              'tool_label':q['tool_label'], 'x':q['x'], 'y':q['y'], 'frame':q['frame']})

    # Output list of dictionaries to pandas dataframe and export to CSV.
    col_order = ['classification_id','user_name','user_id','created_at','subject_ids',
               'tool','tool_label','x','y','frame']
    out = pd.DataFrame(clist)[col_order]
    out.to_csv(markfile_out,index_label='mark_id')
    
class ClassificationData(object):
    def __init__(self, file):
        # Read the data into a table
        self.data = ascii.read(file)
        
        # Save the keys and users as attributes
        self.cols = self.data.keys
        self.users = list(set(self.data['user_name']))

    def users(self):
        """
        Show the user data
        """
        # Print some stats    
        print('Number of users:',len(self.users))
        print('Number of classifications:',len(self.data)/4.)
        

    def objects_of_interest(self):
        """
        ID the tiles as a function of classification
        """
        ids = self.data['subject_ids']
         
    