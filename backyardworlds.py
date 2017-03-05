import sys
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from astropy.io import ascii
from astropy.time import Time
from astropy.table import Table

class ClassificationData(object):
    def __init__(self, classification_file, subject_file, start_date='2017-02-15 12:00:00'):
        """
        Create the ClassificationData object and load the data
        
        Parameters
        ----------
        classification_file: str
            The CSV file containing the classification data
        subject_file: str
            The CSV file containing the subject metadata
        start_date: str
            The dat after which to consider data
        
        """
        # Read the classification data into a table
        self.data = ascii.read(classification_file)
        
        # Convert date string to astropy.time.Time
        dates = [Time(t.replace(' UTC','')) for t in self.data['created_at']]
        
        # Trim data to post-launch
        start = Time(start_date)
        self.data = self.data[np.where(dates>start)]
        
        # Save the keys, users, and subjects as attributes
        self.cols = self.data.colnames
        self.users = list(set(self.data['user_name']))
        
        # Attribute for clicked subjects
        self.clicked = list(np.unique(self.data['subject_ids']))
        
        # Attribute for retired subjects
        self.retired = []
        
        # Read the subject metadata into a table
        self.subjects = ascii.read(subject_file, format='fixed_width')
        
    def get_subject(self, subject_id, plot='composite'):
        """
        Get the classification data for a particular subject
        
        Parameters
        ----------
        subject_id: int
            The id for the subject to look at
        
        Returns
        -------
        astropy.table.Table
            A table of all the data for this subject
        """
        # Just get clicks for this subject
        subject = self.data[self.data['subject_ids']==subject_id]

        # Get subject metadata
        meta = self.subjects[self.subjects['subject_id']==subject_id][0]
        
        if subject:
            # Group by frame
            frames = subject.group_by('frame').groups

            # Get all the click locations
            xy = np.array(subject[['x','y']])

            # Convert the locations to coordinates
            coords = get_coordinates(xy, meta)

            # Find the centers of the sufficiently dense clusters
            clusters = cluster_centers(xy)

            if plot=='composite':

                fig, ax = plt.subplots()

                c = ['b', 'g', 'r', 'm']
                for n, frame in zip(frames.keys['frame'], frames):

                    # Pull out the coordinates
                    xy = np.array(frame[['x','y']])

                    # Plot it!
                    ax.scatter(xy['x'], xy['y'], facecolors='none', 
                                edgecolors=c[n], s=80, alpha=0.3,
                                label='Frame {}'.format(n))

                # Plot the grouping center
                ax.scatter(*clusters.T, marker='+', c='k', s=100, lw=2,
                            label='Centroids')

                # Put RA and Dec on axis ticks
                # xlabels = [item.get_text() for item in ax.get_xticklabels()]
                # ylabels = [float(item.get_text()) for item in ax.get_yticklabels()]
                # labels = np.array([(x,y) for x,y in zip(xlabels,ylabels)], 
                #                   dtype=[('x', '>f4'), ('y', '>f4')])
                # xlabels, ylabels  = get_coordinates(labels, meta)
                # ax.set_xticklabels(labels)

            elif plot:

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

        else:
            print('No classifications for subject',subject_id)
            
    def get_retired(self, retirement=15):
        """
        ID the subjects that are retired
        """
        # Get all the clicked subjects
        clicks = np.array(self.data['subject_ids'])
        
        # Count how many clicks for each subject
        counts = np.bincount(clicks)
        idx = np.nonzero(counts)[0]
        
        # Store the subjects with the appropriate number of clicks
        self.retired = [id for id,n in zip(idx,counts[idx]) if n>=retirement]
        
        print('Retired:',len(self.retired))

def get_coordinates(coords, metadata):
    """
    Calculates the RA and Dec values of the given coords
    based on the RA dn Dec of the tile center
    
    Parameters
    ----------
    coords: array-like
        The (x,y) pixel locations to convert to (RA, Dec)
    metadata: astropy.table.row.Row
        The metadata for the given subject
    
    Returns
    -------
    np.ndarray
        The (Ra,Dec) of the input locations
    """
    # First some constants
    pio180 = np.pi/180.0            # This is PI divided by 180.
    pixrad = (2.75/3600)*pio180     # unWISE pixel, in radians

    #  Next we'll need the ra and dec of center of the tile in question
    # note that use the subtile center for this purpose
    # ratile is the ra of the tile center
    # dectile is the dec of the tile center
    ratile, dectile = [metadata['RA'],metadata['Dec']] or \
        list(map(float,metadata['VizieR'].split('c=')[1].split('+')))
    
    tilename = metadata['images'].split(',')[0]
    
    # Make arrays of the x and y coordinates
    x, y = np.asarray(coords['x']), np.asarray(coords['y'])

    # Here are some sines and cosines for the gnomonic projection
    sinra = np.sin(ratile*pio180)
    sindec = np.sin(dectile*pio180)
    cosra = np.cos(ratile*pio180)
    cosdec = np.cos(dectile*pio180)

    # We'll also need to know subfx, subfy, which tell you where
    # the subtile is in the tile.
    # You can read these from the names of the images
    # for example:  "unwise-1600m197.x3.y6.e0.jpeg"
    # You can read from this that subfx = 3   subfy = 6
    subfx = int(tilename.split('.')[1][1:])
    subfy = int(tilename.split('.')[2][1:])

    # xc,yc will be the coordinates of subtile center in the tangent plane
    xc =-((subfx+0.5)*256.0 - 1024.5)*pixrad
    yc = ((subfy+0.5)*256.0 - 1024.5)*pixrad

    # xp and yp will be the coordinates in the tangent plane
    xp = xc + (256-x)*pixrad/2.0
    yp = yc + (256-y)*pixrad/2.0

    # Here's the Gnomonic projection
    rho = np.sqrt(xp*xp+yp*yp)
    cg = np.arctan(rho)
    dec = np.arcsin(np.cos(cg)*sindec+yp*np.sin(cg)*cosdec/rho)/pio180 
    ra = ratile + np.arctan(xp*np.sin(cg),rho*cosdec*np.cos(cg)-yp*sindec*np.sin(cg))/pio180 

    # now flip around the RA and dec as needed
    dec = [180.0-d if d>90.0 else -180.0-d if d<-90 else d for d in dec]
    ra = [r+180.0 if d>90.0 or d<-90 else r for r,d in zip(ra,dec)]
    ra = [r+360.0 if r<0 else r for r in ra]
    ra = [r%360.0 for r in ra]
    
    return (ra, dec)

def cluster_centers(X, eps=20, min_samples=3):
    """
    Calculate the centers of the point clusters given the
    radius (eps) and minimum number of points (min_samples)
    
    Parameters
    ----------
    X: array-like
        The list of (x,y) coordinates of all clicks
    eps: int
        The distance threshold for cluster membership
    min_samples: int
        The minimum number of points in a cluster to be
        considered interesting
    
    Returns
    -------
    np.ndarray
        An array of the cluster centers
    """
    X = np.array([list(x) for x in X])
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    
    # Group clusters
    clusters = []
    for k in unique_labels:
        class_member_mask = (labels == k)
        xy = X[class_member_mask & core_samples_mask]
        if len(xy)>min_samples:
            clusters.append(xy)
    
    # Get 2D mean of each cluster
    clusters = np.asarray([np.mean(c, axis=0) for c in clusters])
    
    return clusters

def subject_CSV(classfile_in='backyard-worlds-planet-9-subjects.csv', markfile_out='subjects.txt'):
    """
    Generates a readable CSV file from exported Zooniverse data
    
    Parameters
    ----------
    classfile_in: str
        The file to convert
    markfile_out: str
        The path and filename of the output file
    """
    # Read in subject CSV and expand JSON fields
    subjects = pd.read_csv(classfile_in)
    subjects['metadata'] = [json.loads(q) for q in subjects.metadata]
    subjects['locations'] = [json.loads(q) for q in subjects.locations]
    
    # ['!IRSA Finder Chart', 'image3', '!VizieR', 'Tile Number', 'id', 'image0', 'Modified Julian Dates of Each Epoch', 'id numbers of nearest subtiles', 'subtile center', '!SIMBAD', 'image1', 'image2']
    
    # Pull out the metadata for each entry
    out = []
    for index,sub in subjects.iterrows():
        try:
            id = int(sub.subject_id)
            ra = sub.metadata.get('RA')
            dec = sub.metadata.get('dec')
            simbad = sub.metadata.get('!SIMBAD')
            vizier = sub.metadata.get('!VizieR')
            irsa = sub.metadata.get('!IRSA Finder Chart')
            images = ', '.join([sub.metadata.get('image{}'.format(n)) for n in [0,1,2,3]])
            mjd = sub.metadata.get('Modified Julian Dates of Each Epoch')
            center = sub.metadata.get('subtile center')
            tilenum = int(sub.metadata.get('Tile Number'))
            nearest = sub.metadata.get('id numbers of nearest subtiles')
            entry = [id, ra, dec, simbad, vizier, irsa, images, mjd, center, tilenum, nearest]

            if entry not in out:
                out.append(entry)
        except TypeError:
            pass
    
    # Write the data to CSV
    cols = ['subject_id', 'RA', 'Dec', 'SIMBAD', 'VizieR', 'IRSA', 'images', 'MJD', 'Center', 'Tilenum', 'Nearest']
    out = Table(np.array(out), names=cols, dtype=[int,float,float,str,str,str,str,str,str,int,str])
    out.write(markfile_out, format='ascii.fixed_width')
    
def classification_CSV(classfile_in='backyard-worlds-planet-9-classifications.csv', markfile_out='classifications.csv'):
    """
    Generates a readable CSV file from exported Zooniverse data
    
    Parameters
    ----------
    classfile_in: str
        The file to convert
    markfile_out: str
        The path and filename of the output file
    
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
    out.to_csv(markfile_out, index_label='mark_id')
    