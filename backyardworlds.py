import sys
import os
import pandas as pd
import numpy as np
import json
import glob
import warnings
import matplotlib.pyplot as plt
import astropy.coordinates as coords
import astropy.units as q
import astropy.table as at
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn import metrics
from astropy.io import ascii
from astropy.io import fits
from astropy.time import Time
from astropy.table import Table
from astroquery.irsa import Irsa
from astroquery.vizier import Vizier
from astroquery.simbad import Simbad

warnings.filterwarnings("ignore")

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
        print('Loading classifications...')
        #self.data = ascii.read(classification_file, format='csv')
        data_files = glob.glob(classification_file.replace('.csv','*'))
        data = [ascii.read(f, format='fixed_width') for f in data_files]
        self.data = at.vstack(data)
        
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
        print('Loading subjects...')
        #self.subjects = ascii.read(subject_file, format='fixed_width')
        subject_files = glob.glob(subject_file.replace('.txt','*'))
        subjects = [ascii.read(f, format='fixed_width') for f in subject_files]
        self.subjects = at.vstack(subjects)
                
    def get_subject(self, subject_id, radius=20, num=4, plot=True, verbose=False):
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
        
        # Print some stuff
        print('Subject id:',subject_id,'\n')
        print('Users:\n','\n '.join(list(set(subject['user_name']))),'\n')
        
        if subject:
            # Group by frame
            frames = subject.group_by('frame').groups
            
            # Print number of clicks in each group
            for n,f in enumerate(frames):
                user_list = len(list(set(f['user_name'])))
                print('Frame {} has {} clicks from {} users.'.format(n,len(f),user_list))
            print('\n')

            # Get all the click locations
            xy = np.array(subject[['x','y']])

            # Convert the locations to coordinates
            coords = get_coordinates(xy, meta)

            # Find the centers of the sufficiently dense clusters
            clusters, counts = cluster_centers(xy, radius=radius, num=num)
            
            # Search catalogs at cluster center
            results = []
            for idx,(x,y) in enumerate(clusters):
                cluster = np.array([(x,y)], dtype=[('x',float),('y',float)])
                ra, dec = get_coordinates(cluster, meta)
                table = catalog_search(ra[0], dec[0])
                table['#'] = counts[idx]
                results.append(table)
            
            if results:
                results = at.vstack(results)
                results.pprint(max_width=120)

            if plot:

                fig, ax = plt.subplots(figsize=(5,5))

                c = ['b', 'g', 'r', 'm']
                for n, frame in zip(frames.keys['frame'], frames):

                    # Pull out the coordinates
                    xy = np.array(frame[['x','y']])

                    # Plot it!
                    ax.scatter(xy['x'], xy['y'], facecolors='none', 
                                edgecolors=c[n], s=80, alpha=0.3,
                                label='Frame {}'.format(n))
                    
                    ax.set_xlim(0,512)
                    ax.set_ylim(0,512)

                # Plot the grouping center
                try:
                    ax.scatter(*clusters.T, marker='+', c='k', s=100, lw=2,
                               label='Centroids')
                    
                    # Number the clusters for reference
                    for n,(x,y) in enumerate(clusters):
                        plt.annotate(str(n), xy=(x,y), xytext=(x+10, y+10))
                except:
                    pass

                # Put RA and Dec on axis ticks
                # xlabels = [item.get_text() for item in ax.get_xticklabels()]
                # ylabels = [float(item.get_text()) for item in ax.get_yticklabels()]
                # labels = np.array([(x,y) for x,y in zip(xlabels,ylabels)], 
                #                   dtype=[('x', '>f4'), ('y', '>f4')])
                # xlabels, ylabels  = get_coordinates(labels, meta)
                # ax.set_xticklabels(labels)
            
            return subject

        else:
            print('No classifications for subject',subject_id)
            
    def get_retired(self, retirement=15):
        """
        ID the subjects that are retired
        
        Parameters
        ----------
        retirement: int
            The number of clicks necessary to retire a subject
        """
        # Get all the clicked subjects
        clicks = np.array(self.data['subject_ids'])
        
        # Count how many clicks for each subject
        counts = np.bincount(clicks)
        idx = np.nonzero(counts)[0]
        
        # Store the subjects with the appropriate number of clicks
        self.retired = [id for id,n in zip(idx,counts[idx]) if n>=retirement]
        
        print('Retired:',len(self.retired))

def catalog_search(ra, dec, radius=10):
    """
    Search Simbad, 2MASS, and WISE catalogs for object
    
    Parameters
    ----------
    ra: float
        The right ascension
    dec: float
        The declination
    radius: float
        The search radius
        
    Returns
    -------
    astropy.table.Table
        A table of the search results
    """
    c = coords.ICRS(ra=ra*q.deg, dec=dec*q.deg)

    # Query 2MASS and WISE
    WISE = Vizier.query_region(c, radius=radius*q.arcsec, catalog=['II/328/allwise'])
    MASS = Vizier.query_region(c, radius=radius*q.arcsec, catalog=['II/246/out'])
    SIMB = Simbad.query_region(c, radius=(radius+10)*q.arcsec)

    if MASS:
        J, H, K = [MASS[0][m][0] for m in ['Jmag','Hmag','Kmag']]
    else:
        J, H, K = [np.nan]*3
        
    if WISE:
        W1, W2, W3, W4 = [WISE[0][m][0] for m in ['W1mag','W2mag','W3mag','W4mag']]
    else:
        W1, W2, W3, W4 = [np.nan]*4
    
    if SIMB:
        name = SIMB['MAIN_ID'][0]
    else:
        name = 'Not in Simbad'
    
    # result = at.Table(np.array([name, ra, dec, J, H, K, W1, W2, W3, W4]), masked=True, 
    #                names=['name','ra','dec','J','H','K','W1','W2','W3','W4'],
    #                dtype=[str, float, float, float, float, float, float, float, float, float])
    
    result = at.Table(np.array([name, ra, dec, J, H, K, W1, W2]), masked=True, 
                   names=['name','ra','dec','J','H','K','W1','W2'],
                   dtype=[str, float, float, float, float, float, float, float])
    
    return result

def get_coordinates(coords, metadata):
    """
    Calculates the RA and Dec values of the given coords
    based on the RA dn Dec of the tile center
    
    Parameters
    ----------
    coords: array-like
        The (x,y) pixel locations to convert to (RA, Dec)
    metadata: astropy.table.row.Row, dict
        The metadata for the given subject
    
    Returns
    -------
    np.ndarray
        The (Ra,Dec) of the input locations
    """
    # First some constants
    pio180 = np.pi/180.0            # This is PI divided by 180.
    pixrad = (2.75/3600)*pio180     # unWISE pixel, in radians

    # Next we'll need the ra and dec of center of the tile in question
    # note that use the subtile center for this purpose
    # ratile is the ra of the tile center
    # dectile is the dec of the tile center
    ratile, dectile = metadata['RA'], metadata['Dec']
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

def cluster_centers(coords, radius=20, num=3):
    """
    Calculate the centers of the point clusters given the
    radius and minimum number of points
    
    Parameters
    ----------
    coords: array-like
        The list of (x,y) coordinates of all clicks
    radius: int
        The distance threshold for cluster membership
    num: int
        The minimum number of points in a cluster to be
        considered interesting
    
    Returns
    -------
    np.ndarray
        An array of the cluster centers
    """
    coords = np.array([list(x) for x in coords])
    db = DBSCAN(eps=radius, min_samples=num).fit(coords)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    
    # Group clusters
    clusters, counts = [], []
    for k in unique_labels:
        class_member_mask = (labels == k)
        xy = coords[class_member_mask & core_samples_mask]
        clicks = len(xy)
        if clicks>num:
            clusters.append(xy)
            counts.append(clicks)
    
    # Get 2D mean of each cluster
    clusters = np.asarray([np.mean(c, axis=0) for c in clusters])
    
    return clusters, counts

def subject_CSV(classfile_in='/Users/jfilippazzo/Desktop/backyard worlds/backyard-worlds-planet-9-subjects.csv', markfile_out='data/subjects.txt', tile2subtile='data/neo1_meta-atlas.trim.fits'):
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
    print('Subjects:',len(subjects))
    subjects['metadata'] = [json.loads(q) for q in subjects.metadata]
    subjects['locations'] = [json.loads(q) for q in subjects.locations]
    
    # Get the RA and Dec of each tile center
    coord_lookup = fits.getdata(tile2subtile, 1)
    
    # Pull out the metadata for each entry
    clists = []
    clist = []
    for index,sub in subjects.iterrows():
        try:
            id = int(sub.subject_id)
            #ra = sub.metadata.get('RA')
            #dec = sub.metadata.get('dec')
            simbad = sub.metadata.get('!SIMBAD')
            vizier = sub.metadata.get('!VizieR')
            irsa = sub.metadata.get('!IRSA Finder Chart')
            images = ', '.join([sub.metadata.get('image{}'.format(n)) for n in [0,1,2,3]])
            mjd = sub.metadata.get('Modified Julian Dates of Each Epoch')
            center = sub.metadata.get('subtile center')
            tilenum = int(sub.subject_set_id)
            ra, dec = coord_lookup[tilenum]
            nearest = sub.metadata.get('id numbers of nearest subtiles')
            entry = [id, ra, dec, simbad, vizier, irsa, images, mjd, center, tilenum, nearest]

            clist.append(entry)
        
        except TypeError:
            continue
        
        if len(clist)>=50000:
            clists.append(clist)
            clist = []
    
    if clist:
        clists.append(clist)
            
    # Write the data to CSV
    for n,clist in enumerate(clists):
        cols = ['subject_id', 'RA', 'Dec', 'SIMBAD', 'VizieR', 'IRSA', 'images', 'MJD', 'Center', 'Tilenum', 'Nearest']
        out = Table(np.array(clist), names=cols, dtype=[int,float,float,str,str,str,str,str,str,int,str])
        out.write(markfile_out.replace('.txt','{}.txt'.format(n)), format='ascii.fixed_width')
    
def classification_CSV(classfile_in='/Users/jfilippazzo/Desktop/backyard worlds/backyard-worlds-planet-9-classifications.csv', markfile_out='data/classifications.txt'):
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
    print('Classifications:',len(classifications))
    #classifications['metadata_json'] = [json.loads(q) for q in classifications.metadata]
    classifications['annotations_json'] = [json.loads(q) for q in classifications.annotations]
    #classifications['subject_data_json'] = [json.loads(q) for q in classifications.subject_data]

    clists = []
    clist = []
    for index, c in classifications.iterrows():
                
        for q in c.annotations_json:

            try:
                for i in q['value']:

                    try:
                        id = c.classification_id
                        user = c.user_name
                        user_id = 1 #c.user_id
                        created = c.created_at
                        sub_id = c.subject_ids
                        x = float(i.get('x'))
                        y = float(i.get('y'))
                        frame = int(i.get('frame'))
                        entry = [id, user, user_id, created, sub_id, x, y, frame]
                        clist.append(entry)

                    except:
                        continue
            except:
                continue

            if len(clist)>=300000:
                clists.append(clist)
                clist = []
    
    if clist:
        clists.append(clist)

    # Write the data to CSV
    for n,clist in enumerate(clists):
        cols = ['classification_id','user_name','user_id','created_at','subject_ids','x','y','frame']
        out = Table(np.array(clist), names=cols, dtype=[int,str,int,str,int,float,float,int])
        out.write(markfile_out.replace('.txt','{}.txt'.format(n)), format='ascii.fixed_width')    
    