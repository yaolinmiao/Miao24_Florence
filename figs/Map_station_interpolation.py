#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 13:23:28 2021

@author: lviens
"""
 

from scipy.ndimage.filters import gaussian_filter
from cartopy.feature import NaturalEarthFeature, LAND, COASTLINE, ShapelyFeature
from cartopy.io.shapereader import Reader
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import cmocean
from osgeo import gdal
from pandas import read_excel
import pandas as pd


def scale_bar(ax, length=None, location=(0.5, 0.1), linewidth=3):
    """
    ax is the axes to draw the scalebar on.
    length is the length of the scalebar in km.
    location is center of the scalebar in axis coordinates.
    (ie. 0.5 is the middle of the plot)
    linewidth is the thickness of the scalebar.
    """
    #Get the limits of the axis in lat long
    llx0, llx1, lly0, lly1 = ax.get_extent(ccrs.PlateCarree())
    #Make tmc horizontally centred on the middle of the map,
    #vertically at scale bar location
    sbllx = (llx1 + llx0) / 2
    sblly = lly0 + (lly1 - lly0) * location[1]
    tmc = ccrs.TransverseMercator(sbllx, sblly)
    #Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent(tmc)
    #Turn the specified scalebar location into coordinates in metres
    sbx = x0 + (x1 - x0) * location[0]
    sby = y0 + (y1 - y0) * location[1]

    #Calculate a scale bar length if none has been given
    #(Theres probably a more pythonic way of rounding the number but this works)
    if not length: 
        length = (x1 - x0) / 5000 #in km
        ndim = int(np.floor(np.log10(length))) #number of digits in number
        length = round(length, -ndim) #round to 1sf
        #Returns numbers starting with the list
        def scale_number(x):
            if str(x)[0] in ['1', '2', '5']: return int(x)        
            else: return scale_number(x - 10 ** ndim)
        length = scale_number(length) 

    #Generate the x coordinate for the ends of the scalebar
    bar_xs = [sbx - length * 500, sbx + length * 500]
    #Plot the scalebar
    ax.plot(bar_xs, [sby, sby], transform=tmc, color='k', linewidth=linewidth)
    #Plot the scalebar label
    ax.text(sbx, sby+100, str(length) + ' km', transform=tmc,
            horizontalalignment='center', verticalalignment='bottom')


def get_dec(string):
    deg =  string.split('° ' ) 
    dec = deg[1].split("' ") 
    return (float(deg[0]) + float(dec[0])/60  ) * (-1 if dec[1] in ['W', 'S'] else 1) 

 #%%
#Grid of Earth's surface depicting the bedrock underneath the ice sheets. (1-minute resolution)
# Downloaded here: https://www.ncei.noaa.gov/maps/grid-extract/
input = 'OriginalEtopo.tiff'
output = 'bathy1min.nc'
#convert to NetCDF file
ds = gdal.Translate(output, input, format='NetCDF')

# Open ETOPO1 file 
ds = xr.open_dataset('bathy1min.nc')
lon = ds.variables['lon'][:]
lat = ds.variables['lat'][:]
bathy = ds.variables['Band1'][:]
bathy = np.ma.masked_greater_equal(bathy,900)
bathy = np.ma.masked_less_equal(bathy,-1500)


#%%
# Create figure + ax1 fancy
fig = plt.figure(figsize=(10,8),facecolor='w')
ax1 = fig.add_subplot(111, projection=ccrs.PlateCarree())
# Add Cartopy coastines.
north = 44.25
south = 43.5
west = -125.
east = -123.9
ax1.set_extent([west, east, south, north], ccrs.PlateCarree())
ax1.coastlines()
# Add gridlines into the figure. We disabled interior gridline but kept the labels.
gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=2,
color='black',alpha=0.0, linestyle='--')
# Disable top and right grid line labels.
gl.top_labels = gl.right_labels = False
# Add Latitude/Longitude text labels.
ax1.text(-0.08, 0.55, 'Longitude'u' [\N{DEGREE SIGN}]', va='bottom',
ha='center', rotation='vertical', rotation_mode='anchor',
transform=ax1.transAxes)
ax1.text(0.5, -0.075, 'Latitude'u' [\N{DEGREE SIGN}]', va='bottom',
ha='center',rotation='horizontal', rotation_mode='anchor',
transform=ax1.transAxes)

ax1.text(-124.075998, 43.9826, 'Florence',color='k',fontsize=12, bbox=dict(fill=True, \
        edgecolor='black', color='w', linewidth=0,alpha=0.8, zorder=6))
#ax1.plot(-124.0998, 43.9826, color='y', marker='H', linestyle='dashed',linewidth=2, markersize=12,zorder=6)
scale_bar(ax1,length=10)
# Add 1000 300, and 200 meters bathymetry lines and labels.
# We will apply a Gaussian filter to smooth the data.
bathy_levels = [-1000,-300,-100]
Ct = ax1.contour(gaussian_filter(lon,2),gaussian_filter(lat,2)
,gaussian_filter(bathy,2),bathy_levels,colors='black',latlon=True,
linewidths=0.6,linestyles='solid')
manual_locations = [(-124.8,43.6),(-124.7,44),(-124.3,43.6)] 
clbls = ax1.clabel(Ct,fmt='%i', fontsize=9,manual=manual_locations, colors="black")

# Add Topo
lon1 = ds.variables['lon'][:]
lat1 = ds.variables['lat'][:]
topo1 = ds.variables['Band1'][:]
#topo1 = np.ma.masked_less_equal(topo1,-55)
#topo_levels = [100,200]
#Ct2 = ax1.contour(gaussian_filter(lon1,2),gaussian_filter(lat1,2),
#gaussian_filter(topo1,2),topo_levels,colors='white',latlon=True,
#linewidths=0.7,linestyles='solid')


# Create min and max data range (and spacing).
# Plot ETOPO1 data using beautiful cmocean color pallete.

#plot1 = ax1.contourf(lon, lat, topo1, 30, cmap=cmocean.cm.topo,
#transform=ccrs.PlateCarree(),latlon=True,vmin=-3000, vmax=3000,
#extend="both")

plot = ax1.contourf(lon, lat, bathy, 100, cmap=cmocean.cm.topo,
transform=ccrs.PlateCarree(),latlon=True,vmin=-1200, vmax=1200,
extend="both")
          

# Define colobar options (e.g. min/max range, label and where to
# place into the plot).
ticks  = [-1200, -600, 0, 600 ]
cax    = fig.add_axes([0.52, 0.8, 0.2, 0.015])
cb     = fig.colorbar(plot, cax=cax, orientation="horizontal", panchor=(0.5,0.5),shrink=0.3,ticks=ticks)

cb.set_label(r'Relief [m]', fontsize=9, color='0.2',labelpad=0)
cb.ax.tick_params(labelsize=9, length=2, color='0.2', labelcolor='0.2',direction='in')
cb.set_ticks(ticks)


#%%

dist1 = []
dist1.append(6.292)
cable_route_LV =  [ [43.994280, -124.113209],  [43.994197, -124.112762] , [43.995035, -124.111966] , [43.995032, -124.110943], [43.995789, -124.110132] , [43.996337, -124.109208] , [43.997325, -124.108897] , [43.997607, -124.110414] , [43.998171, -124.112275] , [43.997656, -124.115130], [43.997976, -124.116464], [43.997900, -124.117734], [43.998326, -124.118493], [43.998405, -124.119140], [43.999846, -124.119539], [44.000751, -124.120420], [44.002304, -124.120440], [44.003399, -124.121078], [44.004417, -124.122217], [44.004756, -124.122256], [44.005953, -124.121661], [44.009592, -124.122903], [44.011036, -124.122612], [44.012271, -124.122124], [44.017723, -124.123048], [44.020364, -124.122612], [44.022081, -124.123543], [44.023420, -124.123996], [44.024469, -124.124802], [44.026080, -124.126412]  , [44.027329, -124.126639], [44.030043, -124.127872], [44.031038, -124.129004], [44.033271, -124.129216], [44.033626, -124.131898], [44.034128, -124.131941], [44.034158, -124.132510], [44.034703, -124.132652]  ]

latcable = []
loncable = []
for i in np.arange(len(cable_route_LV)):
    latcable.append(cable_route_LV[i][0])
    loncable.append(cable_route_LV[i][1])


file_name = 'RPL_Data_for_AKORN_Fiber.xls'

dfs = read_excel(file_name)
datastart = []
dataend = []
eqMw = []

staname2 = []
for i in np.arange(65, 9, -1):
    df = dfs.iloc[i,:]
    print(df[1], i)
    latcable.append( get_dec(df[1]) )
    loncable.append( get_dec(df[2]) )
    if df[1] =="43° 45.7211' N":
        end_bur = [get_dec(df[2]) , get_dec(df[1])]
    dist1.append(abs(np.float( df[4]) - 1957.293 )+6.292 )
# stop
#%%

latcable.append( 45.260170 )
latcable.append(50.539679 )
loncable.append( -127.781099 )
loncable.append( -134.284349 )

ax1.plot(loncable,latcable, 'k', linewidth = 4)
#%%
print(2500*20)
idx = (np.abs(np.array(dist1) - 50)).argmin()
print(idx)
addidx = len(cable_route_LV)
#ax1.plot(loncable[addidx:addidx+idx],latcable[addidx:addidx+idx], 'orange', linewidth = 4)
ax1.plot(loncable[:addidx+idx],latcable[:addidx+idx], 'orange', linewidth = 4)

fnt =13
OBS =[  -124.549, 43.772] #W
ax1.scatter(OBS[0],OBS[1],s = 100, edgecolors = 'k'  ,color = 'r', linewidth = .1)
ax1.text(OBS[0]-.03,OBS[1]-.04, 'Buoy' , fontsize = fnt+1)
#%%


ax2 = fig.add_axes([0.070, 0.03, 0.37, 0.35], projection=ccrs.PlateCarree())

ds = xr.open_dataset('bathy1min.nc')
lon = ds.variables['lon'][:]
lat = ds.variables['lat'][:]
bathy = ds.variables['Band1'][:]

west = -131
east = -122
south = 41.5
north = 45.35

ax2.set_extent([west, east, south, north], ccrs.PlateCarree())
ax2.coastlines()

ax2.plot(loncable,latcable, 'k', linewidth = 4)
ax2.plot(loncable[:addidx+idx],latcable[:addidx+idx], 'orange', linewidth=4, transform=ccrs.PlateCarree())


plot = ax2.contourf(lon, lat, bathy, 100, cmap=cmocean.cm.topo,
transform=ccrs.PlateCarree(),latlon=True,vmin=-4000, vmax=4000,
extend="both")


# Draw tates.    
states_provinces = NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines',
        scale='50m',facecolor='none')
ax2.add_feature(LAND)
ax2.add_feature(COASTLINE)
ax2.add_feature(states_provinces, edgecolor='k')

ax2.text(-124., 43., 'Oregon',color='k',fontsize=8, bbox=dict(fill=True, \
        edgecolor='black', color='w', linewidth=0,alpha=0.8, zorder=6))
ax2.text(-130., 42., 'Pacific ',color='k',fontsize=8, bbox=dict(fill=True, \
        edgecolor='black', color='w', linewidth=0,alpha=0.8, zorder=6))
ax2.text(-128., 43.7, 'Juan de Fuca',color='k',fontsize=8,rotation=-31, bbox=dict(fill=True, \
        edgecolor='black', color='w', linewidth=0,alpha=0.8, zorder=6))
#ax2.plot(-124.1, 43.1, color='y', marker='H', linestyle='dashed',linewidth=2, markersize=12,zorder=6)

pd.options.display.max_rows = 9999

df = pd.read_csv('query.csv')

lat = df['latitude']
lon = df['longitude']
mag = df['mag']



#db = np.loadtxt('query.csv', skiprows=1,usecols=(2,3,5) )
#la = db[:,0]
#lo = db[:,1]
#mag = db[:,2]
s = [2.5**n for n in mag]

#for lo, la, m in zip(lon, lat, mag):
ax2.scatter(lon,lat, s=s, zorder=5, edgecolors='k',color = 'grey',linewidth = .2, )

fname = 'shp/PB2002_plates.shp'
shape_feature = ShapelyFeature(Reader(fname).geometries(), ccrs.PlateCarree(), facecolor='none')
ax2.add_feature(shape_feature)

#%%
ax3 = fig.add_axes([0.63, 0.22, 0.30, 0.33], projection=ccrs.Mollweide())
#ax3.set_global()

ax3.stock_img()
#ax3.coastlines()

ax3.set_global()

ax3.add_feature(COASTLINE,zorder=4)


db = np.loadtxt('eq65.txt')
lon = db[:,0]
lat = db[:,1]
mag = db[:,2]

#nmag = np.ma.masked_less_equal(np.array(mag),6.5)
#mask = np.ma.getmask(nmag)
#nlon = np.ma.masked_array(np.array(lon), nmag.mask)
#nlat = np.ma.masked_array(np.array(lat), nmag.mask)#

s = 2.**mag

#for a, b, c in zip(lon, lat, mag):
#    if c >= 6.5:
#        print (a,b,c)


ax3.scatter(lon,lat, s=s, zorder=5, edgecolors='k',color = 'grey',linewidth = .2, transform=ccrs.PlateCarree())

#%%
west = -124.15
east = -124.085
south = 43.98
north = 44.04
import cartopy.io.img_tiles as cimgt



imagery = cimgt.OSM()
imagery =cimgt.GoogleTiles()
ax4 = fig.add_axes([0.76, 0.65, 0.205, 0.2], projection=imagery.crs)

ax4.set_extent([west, east, south, north], )
#ax4.coastlines()
ax4.add_image(imagery, 14,interpolation='spline36',)# regrid_shape=2000)#, regrid_shape=2000)
ax4.plot(loncable[:addidx+idx],latcable[:addidx+idx], 'orange', linewidth=2, transform=ccrs.PlateCarree())

ax4.scatter(-124.1094, 43.98809, s = 100, marker='v',edgecolors = 'k'  ,color = 'r', transform=ccrs.PlateCarree())
ax4.text(-124.1094-0.038, 43.98809-.001, 'UW.FLRE' , transform=ccrs.PlateCarree())
scale_bar(ax4,length=1,  location=(0.85, 0.85), linewidth=3)

#%%


# Save figure.
plt.savefig('F2.png', transparent=True, bbox_inches = 'tight',pad_inches=0.1, dpi=300)







#%%

#fold = '/Users/lviens/Documents/DAS/Florence/Cable_info/

#ax = plt.axes(projection=ccrs.Mollweide())

#source_proj = ccrs.PlateCarree()
#fname = os.path.join('path_to_natural_earth', 'NE1_50M_SR_W.tif')
#ax.imshow(imread(fname), origin='upper', transform=source_proj,
#          extent=[-180, 180, -90, 90])

plt.show()


