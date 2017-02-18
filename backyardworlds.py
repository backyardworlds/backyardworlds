import sys
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from astropy.io import ascii
from astropy.time import Time

class ClassificationData(object):
    def __init__(self, file, start_date='2017-02-15 12:00:00'):
        # Read the data into a table
        self.data = ascii.read(file)
        
        # Convert date string to astropy.time.Time
        dates = [Time(t.replace(' UTC','')) for t in self.data['created_at']]
        
        # Trim data to post-launch
        start = Time(start_date)
        self.data = self.data[np.where(dates>start)][:5000]

        # Save the keys, users, and subjects as attributes
        self.cols = self.data.colnames
        self.users = list(set(self.data['user_name']))
        self.subjects = list(set(self.data['subject_ids']))

        # Attribute for retired subjects
        self.retired = []
        
    def get_subject(self, subject_id, plot=True):
        """
        Get the classification records for a particular subject
        """
        # 
        subject = self.data[self.data['subject_ids']==subject_id]
                
        if plot:
            # Group by frame
            frames = subject.group_by('frame').groups
        
            # Draw figure
            f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
            
            # Populate the frames
            axes = [ax1,ax2,ax3,ax4]
            for n, frame in zip(frames.keys['frame'], frames):
                
                # Pull out the coordinates
                xy = np.array(frame[['x','y']])
    
                # Plot it!
                axes[n].scatter(xy['x'], xy['y'])
                
                # Plot the cutouts too?!
                
            # Add labels
            for n in range(4):
                axes[n].set_title('Frame {}'.format(n))
                
        return subject
        
    def get_retired(self, retirement=15):
        """
        ID the subjects that are retired
        """
        retire = []
        for sub in self.subjects:
            subject = self.get_subject(sub, plot=False)
            
            class_ids = list(set(subject['classification_id']))
            
            if len(class_ids)>=retirement:
                retire.append(sub)
        
        self.retired = retire
        
        # # Group by subject_id
        # grouped = self.data.group_by('subject_ids').groups
        #
        # # Count how many classifications for each subject
        # counted = np.diff(grouped.indices)
        #
        # # Get indices of those that make cutoff
        # filtered = grouped[np.where(counted>=retirement)]
        #
        # # Get ids of retired subjects
        # self.retired = filtered.groups.keys
        #
        
        
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
    