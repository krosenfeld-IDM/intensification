""" SIACalendar.py

This script is an attempt at an easily updatable, spatially well
resolved (i.e. with as much target LGA population as known) sia calendar
for Nigeria. Primary source is the WHO excel spreadsheet found here:
https://immunizationdata.who.int/ 

This file produces the CSV for campaigns used throughout the repo. """
import os
import sys

## Standard tools
import numpy as np
import pandas as pd

## Compute approximate weights over time based on estimated births
## in the demographics analyses. This is to allocate national level
## estimates of sia doses to the state level.
births = pd.read_csv(os.path.join("_data",
                     "monthly_births_by_state.csv"),
                     index_col=0,parse_dates=["time"])\
                    .set_index(["state","time"])["avg"]
births = births.groupby(["state",lambda t: t[1].year]).mean().unstack()
births = births.T.reindex(np.arange(2000,2024))
births = births.fillna(method="ffill").fillna(method="bfill").T

## Get the region to state mapping for reference as well
s_and_r = pd.read_csv(os.path.join("_data",
                      "states_and_regions.csv"),
                    index_col=0)
s_and_r["macro_region"] = s_and_r["region"].apply(lambda s: s.split()[0])

## 2005 catch up campaign (North only)
## Comment: "50% population (20 northern states)"
total_reach = 28538974
state_dist = births.loc[s_and_r.loc[s_and_r["macro_region"] == "north","state"],
                        np.arange(2000,2005)].sum(axis=1)
state_dist = np.round(total_reach*(state_dist/(state_dist.sum()))).astype(int)
sia_calendar = [("niger","2005-12-6","2005-12-10","9M-15Y",state_dist.loc["niger"]),
                ("abuja","2005-12-6","2005-12-10","9M-15Y",state_dist.loc["abuja"]),
                ("kogi","2005-12-6","2005-12-10","9M-15Y",state_dist.loc["kogi"]),
                ("kwara","2005-12-6","2005-12-10","9M-15Y",state_dist.loc["kwara"]),
                ("plateau","2005-12-6","2005-12-10","9M-15Y",state_dist.loc["plateau"]),
                ("benue","2005-12-6","2005-12-10","9M-15Y",state_dist.loc["benue"]),
                ("nasarawa","2005-12-6","2005-12-10","9M-15Y",state_dist.loc["nasarawa"]),
                ("gombe","2005-12-6","2005-12-10","9M-15Y",state_dist.loc["gombe"]),
                ("yobe","2005-12-6","2005-12-10","9M-15Y",state_dist.loc["yobe"]),
                ("borno","2005-12-6","2005-12-10","9M-15Y",state_dist.loc["borno"]),
                ("bauchi","2005-12-6","2005-12-10","9M-15Y",state_dist.loc["bauchi"]),
                ("taraba","2005-12-6","2005-12-10","9M-15Y",state_dist.loc["taraba"]),
                ("adamawa","2005-12-6","2005-12-10","9M-15Y",state_dist.loc["adamawa"]),
                ("sokoto","2005-12-6","2005-12-10","9M-15Y",state_dist.loc["sokoto"]),
                ("kebbi","2005-12-6","2005-12-10","9M-15Y",state_dist.loc["kebbi"]),
                ("katsina","2005-12-6","2005-12-10","9M-15Y",state_dist.loc["katsina"]),
                ("kano","2005-12-6","2005-12-10","9M-15Y",state_dist.loc["kano"]),
                ("kaduna","2005-12-6","2005-12-10","9M-15Y",state_dist.loc["kaduna"]),
                ("zamfara","2005-12-6","2005-12-10","9M-15Y",state_dist.loc["zamfara"]),
                ("jigawa","2005-12-6","2005-12-10","9M-15Y",state_dist.loc["jigawa"])]

## 2006 catch up campaign (South only)
## Comment: "50% population 17 States in Nigeria"
total_reach = 26353793
state_dist = births.loc[s_and_r.loc[s_and_r["macro_region"] == "south","state"],
                        np.arange(2001,2006)].sum(axis=1)
state_dist = np.round(total_reach*(state_dist/(state_dist.sum()))).astype(int)
sia_calendar += [("imo","2006-10-3","2006-10-9","9M-15Y",state_dist.loc["imo"]),
                 ("anambra","2006-10-3","2006-10-9","9M-15Y",state_dist.loc["anambra"]),
                 ("ebonyi","2006-10-3","2006-10-9","9M-15Y",state_dist.loc["ebonyi"]),
                 ("abia","2006-10-3","2006-10-9","9M-15Y",state_dist.loc["abia"]),
                 ("enugu","2006-10-3","2006-10-9","9M-15Y",state_dist.loc["enugu"]),
                 ("delta","2006-10-3","2006-10-9","9M-15Y",state_dist.loc["delta"]),
                 ("cross river","2006-10-3","2006-10-9","9M-15Y",state_dist.loc["cross river"]),
                 ("bayelsa","2006-10-3","2006-10-9","9M-15Y",state_dist.loc["bayelsa"]),
                 ("rivers","2006-10-3","2006-10-9","9M-15Y",state_dist.loc["rivers"]),
                 ("akwa ibom","2006-10-3","2006-10-9","9M-15Y",state_dist.loc["akwa ibom"]),
                 ("edo","2006-10-3","2006-10-9","9M-15Y",state_dist.loc["edo"]),
                 ("lagos","2006-10-3","2006-10-9","9M-15Y",state_dist.loc["lagos"]),
                 ("ekiti","2006-10-3","2006-10-9","9M-15Y",state_dist.loc["ekiti"]),
                 ("ondo","2006-10-3","2006-10-9","9M-15Y",state_dist.loc["ondo"]),
                 ("osun","2006-10-3","2006-10-9","9M-15Y",state_dist.loc["osun"]),
                 ("oyo","2006-10-3","2006-10-9","9M-15Y",state_dist.loc["oyo"]),
                 ("ogun","2006-10-3","2006-10-9","9M-15Y",state_dist.loc["ogun"])]

## 2007 mop up campaigns (South only)
## Comment: "S. East, S. South & S. West" (1/1/07 to ?)
##          "SW State"                    (3/26/07 to 3/31/07)
## So, the second entry makes it seem like one state was delayed? 
## But as of now, we don't know which one.
total_reach = 2308527+517410
state_dist = births.loc[s_and_r.loc[s_and_r["macro_region"] == "south","state"],
                        np.arange(2002,2007)].sum(axis=1)
state_dist = np.round(total_reach*(state_dist/(state_dist.sum()))).astype(int)
sia_calendar += [("imo","2007-1-1","2007-1-6","9-59M",state_dist.loc["imo"]),
                 ("anambra","2007-1-1","2007-1-6","9-59M",state_dist.loc["anambra"]),
                 ("ebonyi","2007-1-1","2007-1-6","9-59M",state_dist.loc["ebonyi"]),
                 ("abia","2007-1-1","2007-1-6","9-59M",state_dist.loc["abia"]),
                 ("enugu","2007-1-1","2007-1-6","9-59M",state_dist.loc["enugu"]),
                 ("delta","2007-1-1","2007-1-6","9-59M",state_dist.loc["delta"]),
                 ("cross river","2007-1-1","2007-1-6","9-59M",state_dist.loc["cross river"]),
                 ("bayelsa","2007-1-1","2007-1-6","9-59M",state_dist.loc["bayelsa"]),
                 ("rivers","2007-1-1","2007-1-6","9-59M",state_dist.loc["rivers"]),
                 ("akwa ibom","2007-1-1","2007-1-6","9-59M",state_dist.loc["akwa ibom"]),
                 ("edo","2007-1-1","2007-1-6","9-59M",state_dist.loc["edo"]),
                 ("lagos","2007-1-1","2007-1-6","9-59M",state_dist.loc["lagos"]),
                 ("ekiti","2007-1-1","2007-1-6","9-59M",state_dist.loc["ekiti"]),
                 ("ondo","2007-1-1","2007-1-6","9-59M",state_dist.loc["ondo"]),
                 ("osun","2007-1-1","2007-1-6","9-59M",state_dist.loc["osun"]),
                 ("oyo","2007-1-1","2007-1-6","9-59M",state_dist.loc["oyo"]),
                 ("ogun","2007-1-1","2007-1-6","9-59M",state_dist.loc["ogun"])]

## 2008 national follow up campaign
## It has a long time range so it was probably conducted
## at different times in different states, but that's not clear.
## As indicated, it's for the whole country.
total_reach = 28848102
state_dist = births[np.arange(2003,2008)].sum(axis=1).copy()
state_dist = np.round(total_reach*(state_dist/(state_dist.sum()))).astype(int)
sia_calendar += [(s,"2008-11-26","2008-12-15","9-59M",state_dist.loc[s]) for s in state_dist.index]

## 2011 follow up campaign
## Comment: "Northern States 26 to 30 Jan. Southern  States 23 to 27 Feb."
total_reach = 28435589
state_dist = births[np.arange(2006,2011)].sum(axis=1).copy()
state_dist = np.round(total_reach*(state_dist/(state_dist.sum()))).astype(int)
sia_calendar += [("niger","2011-01-26","2011-01-30","9-59M",state_dist.loc["niger"]),
                 ("abuja","2011-01-26","2011-01-30","9-59M",state_dist.loc["abuja"]),
                 ("kogi","2011-01-26","2011-01-30","9-59M",state_dist.loc["kogi"]),
                 ("kwara","2011-01-26","2011-01-30","9-59M",state_dist.loc["kwara"]),
                 ("plateau","2011-01-26","2011-01-30","9-59M",state_dist.loc["plateau"]),
                 ("benue","2011-01-26","2011-01-30","9-59M",state_dist.loc["benue"]),
                 ("nasarawa","2011-01-26","2011-01-30","9-59M",state_dist.loc["nasarawa"]),
                 ("gombe","2011-01-26","2011-01-30","9-59M",state_dist.loc["gombe"]),
                 ("yobe","2011-01-26","2011-01-30","9-59M",state_dist.loc["yobe"]),
                 ("borno","2011-01-26","2011-01-30","9-59M",state_dist.loc["borno"]),
                 ("bauchi","2011-01-26","2011-01-30","9-59M",state_dist.loc["bauchi"]),
                 ("taraba","2011-01-26","2011-01-30","9-59M",state_dist.loc["taraba"]),
                 ("adamawa","2011-01-26","2011-01-30","9-59M",state_dist.loc["adamawa"]),
                 ("sokoto","2011-01-26","2011-01-30","9-59M",state_dist.loc["sokoto"]),
                 ("kebbi","2011-01-26","2011-01-30","9-59M",state_dist.loc["kebbi"]),
                 ("katsina","2011-01-26","2011-01-30","9-59M",state_dist.loc["katsina"]),
                 ("kano","2011-01-26","2011-01-30","9-59M",state_dist.loc["kano"]),
                 ("kaduna","2011-01-26","2011-01-30","9-59M",state_dist.loc["kaduna"]),
                 ("zamfara","2011-01-26","2011-01-30","9-59M",state_dist.loc["zamfara"]),
                 ("jigawa","2011-01-26","2011-01-30","9-59M",state_dist.loc["jigawa"]),
                 ("imo","2011-02-23","2011-02-27","9-59M",state_dist.loc["imo"]),
                 ("anambra","2011-02-23","2011-02-27","9-59M",state_dist.loc["anambra"]),
                 ("ebonyi","2011-02-23","2011-02-27","9-59M",state_dist.loc["ebonyi"]),
                 ("abia","2011-02-23","2011-02-27","9-59M",state_dist.loc["abia"]),
                 ("enugu","2011-02-23","2011-02-27","9-59M",state_dist.loc["enugu"]),
                 ("delta","2011-02-23","2011-02-27","9-59M",state_dist.loc["delta"]),
                 ("cross river","2011-02-23","2011-02-27","9-59M",state_dist.loc["cross river"]),
                 ("bayelsa","2011-02-23","2011-02-27","9-59M",state_dist.loc["bayelsa"]),
                 ("rivers","2011-02-23","2011-02-27","9-59M",state_dist.loc["rivers"]),
                 ("akwa ibom","2011-02-23","2011-02-27","9-59M",state_dist.loc["akwa ibom"]),
                 ("edo","2011-02-23","2011-02-27","9-59M",state_dist.loc["edo"]),
                 ("lagos","2011-02-23","2011-02-27","9-59M",state_dist.loc["lagos"]),
                 ("ekiti","2011-02-23","2011-02-27","9-59M",state_dist.loc["ekiti"]),
                 ("ondo","2011-02-23","2011-02-27","9-59M",state_dist.loc["ondo"]),
                 ("osun","2011-02-23","2011-02-27","9-59M",state_dist.loc["osun"]),
                 ("oyo","2011-02-23","2011-02-27","9-59M",state_dist.loc["oyo"]),
                 ("ogun","2011-02-23","2011-02-27","9-59M",state_dist.loc["ogun"])]

## 2013 follow up campaigns
## 10-5 to 10-9 for North, 4-13 to 4-16 outbreak response (unknown loc)
## and 11-2 to 11-6 for South.
total_reach_north = 17004058
total_reach_south = 13575608
state_dist_north = births.loc[s_and_r.loc[s_and_r["macro_region"] == "north","state"],np.arange(2008,2013)].sum(axis=1)
state_dist_south = births.loc[s_and_r.loc[s_and_r["macro_region"] == "south","state"],np.arange(2008,2013)].sum(axis=1)
state_dist_north = np.round(total_reach_north*(state_dist_north/(state_dist_north.sum()))).astype(int)
state_dist_south = np.round(total_reach_south*(state_dist_south/(state_dist_south.sum()))).astype(int)
state_dist = pd.concat([state_dist_south,state_dist_north],axis=0)
sia_calendar += [("niger","2013-10-5","2013-10-9","9-59M",state_dist.loc["niger"]),
                 ("abuja","2013-10-5","2013-10-9","9-59M",state_dist.loc["abuja"]),
                 ("kogi","2013-10-5","2013-10-9","9-59M",state_dist.loc["kogi"]),
                 ("kwara","2013-10-5","2013-10-9","9-59M",state_dist.loc["kwara"]),
                 ("plateau","2013-10-5","2013-10-9","9-59M",state_dist.loc["plateau"]),
                 ("benue","2013-10-5","2013-10-9","9-59M",state_dist.loc["benue"]),
                 ("nasarawa","2013-10-5","2013-10-9","9-59M",state_dist.loc["nasarawa"]),
                 ("gombe","2013-10-5","2013-10-9","9-59M",state_dist.loc["gombe"]),
                 ("yobe","2013-10-5","2013-10-9","9-59M",state_dist.loc["yobe"]),
                 ("borno","2013-10-5","2013-10-9","9-59M",state_dist.loc["borno"]),
                 ("bauchi","2013-10-5","2013-10-9","9-59M",state_dist.loc["bauchi"]),
                 ("taraba","2013-10-5","2013-10-9","9-59M",state_dist.loc["taraba"]),
                 ("adamawa","2013-10-5","2013-10-9","9-59M",state_dist.loc["adamawa"]),
                 ("sokoto","2013-10-5","2013-10-9","9-59M",state_dist.loc["sokoto"]),
                 ("kebbi","2013-10-5","2013-10-9","9-59M",state_dist.loc["kebbi"]),
                 ("katsina","2013-10-5","2013-10-9","9-59M",state_dist.loc["katsina"]),
                 ("kano","2013-10-5","2013-10-9","9-59M",state_dist.loc["kano"]),
                 ("kaduna","2013-10-5","2013-10-9","9-59M",state_dist.loc["kaduna"]),
                 ("zamfara","2013-10-5","2013-10-9","9-59M",state_dist.loc["zamfara"]),
                 ("jigawa","2013-10-5","2013-10-9","9-59M",state_dist.loc["jigawa"]),
                 ("imo","2013-11-2","2013-11-6","9-59M",state_dist.loc["imo"]),
                 ("anambra","2013-11-2","2013-11-6","9-59M",state_dist.loc["anambra"]),
                 ("ebonyi","2013-11-2","2013-11-6","9-59M",state_dist.loc["ebonyi"]),
                 ("abia","2013-11-2","2013-11-6","9-59M",state_dist.loc["abia"]),
                 ("enugu","2013-11-2","2013-11-6","9-59M",state_dist.loc["enugu"]),
                 ("delta","2013-11-2","2013-11-6","9-59M",state_dist.loc["delta"]),
                 ("cross river","2013-11-2","2013-11-6","9-59M",state_dist.loc["cross river"]),
                 ("bayelsa","2013-11-2","2013-11-6","9-59M",state_dist.loc["bayelsa"]),
                 ("rivers","2013-11-2","2013-11-6","9-59M",state_dist.loc["rivers"]),
                 ("akwa ibom","2013-11-2","2013-11-6","9-59M",state_dist.loc["akwa ibom"]),
                 ("edo","2013-11-2","2013-11-6","9-59M",state_dist.loc["edo"]),
                 ("lagos","2013-11-2","2013-11-6","9-59M",state_dist.loc["lagos"]),
                 ("ekiti","2013-11-2","2013-11-6","9-59M",state_dist.loc["ekiti"]),
                 ("ondo","2013-11-2","2013-11-6","9-59M",state_dist.loc["ondo"]),
                 ("osun","2013-11-2","2013-11-6","9-59M",state_dist.loc["osun"]),
                 ("oyo","2013-11-2","2013-11-6","9-59M",state_dist.loc["oyo"]),
                 ("ogun","2013-11-2","2013-11-6","9-59M",state_dist.loc["ogun"])]

## 2015 follow up campaign in the North
## Comment: "Northern states"
total_reach = 24069024
state_dist = births.loc[s_and_r.loc[s_and_r["macro_region"] == "north","state"],np.arange(2010,2015)].sum(axis=1)
state_dist = np.round(total_reach*(state_dist/(state_dist.sum()))).astype(int)
sia_calendar += [("niger","2015-11-21","2015-11-25","6M-10Y",state_dist.loc["niger"]),
                 ("abuja","2015-11-21","2015-11-25","6M-10Y",state_dist.loc["abuja"]),
                 ("kogi","2015-11-21","2015-11-25","6M-10Y",state_dist.loc["kogi"]),
                 ("kwara","2015-11-21","2015-11-25","6M-10Y",state_dist.loc["kwara"]),
                 ("plateau","2015-11-21","2015-11-25","6M-10Y",state_dist.loc["plateau"]),
                 ("benue","2015-11-21","2015-11-25","6M-10Y",state_dist.loc["benue"]),
                 ("nasarawa","2015-11-21","2015-11-25","6M-10Y",state_dist.loc["nasarawa"]),
                 ("gombe","2015-11-21","2015-11-25","6M-10Y",state_dist.loc["gombe"]),
                 ("yobe","2015-11-21","2015-11-25","6M-10Y",state_dist.loc["yobe"]),
                 ("borno","2015-11-21","2015-11-25","6M-10Y",state_dist.loc["borno"]),
                 ("bauchi","2015-11-21","2015-11-25","6M-10Y",state_dist.loc["bauchi"]),
                 ("taraba","2015-11-21","2015-11-25","6M-10Y",state_dist.loc["taraba"]),
                 ("adamawa","2015-11-21","2015-11-25","6M-10Y",state_dist.loc["adamawa"]),
                 ("sokoto","2015-11-21","2015-11-25","6M-10Y",state_dist.loc["sokoto"]),
                 ("kebbi","2015-11-21","2015-11-25","6M-10Y",state_dist.loc["kebbi"]),
                 ("katsina","2015-11-21","2015-11-25","6M-10Y",state_dist.loc["katsina"]),
                 ("kano","2015-11-21","2015-11-25","6M-10Y",state_dist.loc["kano"]),
                 ("kaduna","2015-11-21","2015-11-25","6M-10Y",state_dist.loc["kaduna"]),
                 ("zamfara","2015-11-21","2015-11-25","6M-10Y",state_dist.loc["zamfara"]),
                 ("jigawa","2015-11-21","2015-11-25","6M-10Y",state_dist.loc["jigawa"])]

## 2016 follow up campaign in South
## Comment: "Phase 2 (Phase 1 in 2015): Southern states (17)
##           Total target (2015+2016 campaigns): 39,160,634
##           JRF 2017: reached 19,102,223 - %cov: 95"
total_reach = 19065787
state_dist = births.loc[s_and_r.loc[s_and_r["macro_region"] == "south","state"],np.arange(2011,2016)].sum(axis=1)
state_dist = np.round(total_reach*(state_dist/(state_dist.sum()))).astype(int)
sia_calendar += [("imo","2016-01-28","2016-02-06","9-59M",state_dist.loc["imo"]),
                 ("anambra","2016-01-28","2016-02-06","9-59M",state_dist.loc["anambra"]),
                 ("ebonyi","2016-01-28","2016-02-06","9-59M",state_dist.loc["ebonyi"]),
                 ("abia","2016-01-28","2016-02-06","9-59M",state_dist.loc["abia"]),
                 ("enugu","2016-01-28","2016-02-06","9-59M",state_dist.loc["enugu"]),
                 ("delta","2016-01-28","2016-02-06","9-59M",state_dist.loc["delta"]),
                 ("cross river","2016-01-28","2016-02-06","9-59M",state_dist.loc["cross river"]),
                 ("bayelsa","2016-01-28","2016-02-06","9-59M",state_dist.loc["bayelsa"]),
                 ("rivers","2016-01-28","2016-02-06","9-59M",state_dist.loc["rivers"]),
                 ("akwa ibom","2016-01-28","2016-02-06","9-59M",state_dist.loc["akwa ibom"]),
                 ("edo","2016-01-28","2016-02-06","9-59M",state_dist.loc["edo"]),
                 ("lagos","2016-01-28","2016-02-06","9-59M",state_dist.loc["lagos"]),
                 ("ekiti","2016-01-28","2016-02-06","9-59M",state_dist.loc["ekiti"]),
                 ("ondo","2016-01-28","2016-02-06","9-59M",state_dist.loc["ondo"]),
                 ("osun","2016-01-28","2016-02-06","9-59M",state_dist.loc["osun"]),
                 ("oyo","2016-01-28","2016-02-06","9-59M",state_dist.loc["oyo"]),
                 ("ogun","2016-01-28","2016-02-06","9-59M",state_dist.loc["ogun"])]

## 2017 follow up campaign
## Not comments, but assumption is similar structure
## as the 2015-16 campaign.
## So I'm assuming the total reach is similar to the comment in the
## 2016 campaign (i.e. the total north + south, with the south listed
## seperately as the 2018 campaign).
total_reach = 40044875-16955354
state_dist = births.loc[s_and_r.loc[s_and_r["macro_region"] == "north","state"],np.arange(2012,2017)].sum(axis=1)
state_dist = np.round(total_reach*(state_dist/(state_dist.sum()))).astype(int)
sia_calendar += [("niger","2017-11-09","2017-11-14","9-59M",state_dist.loc["niger"]),
                 ("abuja","2017-11-09","2017-11-14","9-59M",state_dist.loc["abuja"]),
                 ("kogi","2017-11-09","2017-11-14","9-59M",state_dist.loc["kogi"]),
                 ("kwara","2017-11-09","2017-11-14","9-59M",state_dist.loc["kwara"]),
                 ("plateau","2017-11-09","2017-11-14","9-59M",state_dist.loc["plateau"]),
                 ("benue","2017-11-09","2017-11-14","9-59M",state_dist.loc["benue"]),
                 ("nasarawa","2017-11-09","2017-11-14","9-59M",state_dist.loc["nasarawa"]),
                 ("gombe","2017-11-09","2017-11-14","9-59M",state_dist.loc["gombe"]),
                 ("yobe","2017-11-09","2017-11-14","9-59M",state_dist.loc["yobe"]),
                 ("borno","2017-11-09","2017-11-14","9-59M",state_dist.loc["borno"]),
                 ("bauchi","2017-11-09","2017-11-14","9-59M",state_dist.loc["bauchi"]),
                 ("taraba","2017-11-09","2017-11-14","9-59M",state_dist.loc["taraba"]),
                 ("adamawa","2017-11-09","2017-11-14","9-59M",state_dist.loc["adamawa"]),
                 ("sokoto","2017-11-09","2017-11-14","9-59M",state_dist.loc["sokoto"]),
                 ("kebbi","2017-11-09","2017-11-14","9-59M",state_dist.loc["kebbi"]),
                 ("katsina","2017-11-09","2017-11-14","9-59M",state_dist.loc["katsina"]),
                 ("kano","2017-11-09","2017-11-14","9-59M",state_dist.loc["kano"]),
                 ("kaduna","2017-11-09","2017-11-14","9-59M",state_dist.loc["kaduna"]),
                 ("zamfara","2017-11-09","2017-11-14","9-59M",state_dist.loc["zamfara"]),
                 ("jigawa","2017-11-09","2017-11-14","9-59M",state_dist.loc["jigawa"])] 

## 2017 outbreak response
## Comment: "OBR (Adamawa, Borno and Yobe states)"
total_reach = 4675876
state_dist = births.loc[["adamawa","borno","yobe"],np.arange(2012,2017)].sum(axis=1)
state_dist = np.round(total_reach*(state_dist/(state_dist.sum()))).astype(int)
sia_calendar += [("adamawa","2017-01-12","2017-02-06","6M-10Y",state_dist.loc["adamawa"]),
                 ("borno","2017-01-12","2017-02-06","6M-10Y",state_dist.loc["borno"]),
                 ("yobe","2017-01-12","2017-02-06","6M-10Y",state_dist.loc["yobe"])]

## 2018 follow up campaign in South
## No comments, but assumption is that it's structurally 
## similar to the 2015-16 campaign.
total_reach = 16955354
state_dist = births.loc[s_and_r.loc[s_and_r["macro_region"] == "south","state"],np.arange(2013,2018)].sum(axis=1)
state_dist = np.round(total_reach*(state_dist/(state_dist.sum()))).astype(int)
sia_calendar += [("imo","2018-03-01","2018-03-30","9-59M",state_dist.loc["imo"]),
                 ("anambra","2018-03-01","2018-03-30","9-59M",state_dist.loc["anambra"]),
                 ("ebonyi","2018-03-01","2018-03-30","9-59M",state_dist.loc["ebonyi"]),
                 ("abia","2018-03-01","2018-03-30","9-59M",state_dist.loc["abia"]),
                 ("enugu","2018-03-01","2018-03-30","9-59M",state_dist.loc["enugu"]),
                 ("delta","2018-03-01","2018-03-30","9-59M",state_dist.loc["delta"]),
                 ("cross river","2018-03-01","2018-03-30","9-59M",state_dist.loc["cross river"]),
                 ("bayelsa","2018-03-01","2018-03-30","9-59M",state_dist.loc["bayelsa"]),
                 ("rivers","2018-03-01","2018-03-30","9-59M",state_dist.loc["rivers"]),
                 ("akwa ibom","2018-03-01","2018-03-30","9-59M",state_dist.loc["akwa ibom"]),
                 ("edo","2018-03-01","2018-03-30","9-59M",state_dist.loc["edo"]),
                 ("lagos","2018-03-01","2018-03-30","9-59M",state_dist.loc["lagos"]),
                 ("ekiti","2018-03-01","2018-03-30","9-59M",state_dist.loc["ekiti"]),
                 ("ondo","2018-03-01","2018-03-30","9-59M",state_dist.loc["ondo"]),
                 ("osun","2018-03-01","2018-03-30","9-59M",state_dist.loc["osun"]),
                 ("oyo","2018-03-01","2018-03-30","9-59M",state_dist.loc["oyo"]),
                 ("ogun","2018-03-01","2018-03-30","9-59M",state_dist.loc["ogun"])]

## 2019 follow up campaign in the North
## Comment: "Kano State: 31/10-05/11 - Nasawara State: 30/11 - 
##           Kogi State: 09/12 - Rest of North: 16/11"
total_reach = 21417932
state_dist = births.loc[s_and_r.loc[s_and_r["macro_region"] == "north","state"],np.arange(2014,2019)].sum(axis=1)
state_dist = np.round(total_reach*(state_dist/(state_dist.sum()))).astype(int)
sia_calendar += [("niger","2019-11-16","2019-11-21","9-59M",state_dist.loc["niger"]),
                 ("abuja","2019-11-16","2019-11-21","9-59M",state_dist.loc["abuja"]),
                 ("kogi","2019-12-09","2019-12-15","9-59M",state_dist.loc["kogi"]),
                 ("kwara","2019-11-16","2019-11-21","9-59M",state_dist.loc["kwara"]),
                 ("plateau","2019-11-16","2019-11-21","9-59M",state_dist.loc["plateau"]),
                 ("benue","2019-11-16","2019-11-21","9-59M",state_dist.loc["benue"]),
                 ("nasarawa","2019-11-30","2019-12-05","9-59M",state_dist.loc["nasarawa"]),
                 ("gombe","2019-11-16","2019-11-21","9-59M",state_dist.loc["gombe"]),
                 ("yobe","2019-11-16","2019-11-21","9-59M",state_dist.loc["yobe"]),
                 ("borno","2019-11-16","2019-11-21","9-59M",state_dist.loc["borno"]),
                 ("bauchi","2019-11-16","2019-11-21","9-59M",state_dist.loc["bauchi"]),
                 ("taraba","2019-11-16","2019-11-21","9-59M",state_dist.loc["taraba"]),
                 ("adamawa","2019-11-16","2019-11-21","9-59M",state_dist.loc["adamawa"]),
                 ("sokoto","2019-11-16","2019-11-21","9-59M",state_dist.loc["sokoto"]),
                 ("kebbi","2019-11-16","2019-11-21","9-59M",state_dist.loc["kebbi"]),
                 ("katsina","2019-11-16","2019-11-21","9-59M",state_dist.loc["katsina"]),
                 ("kano","2019-10-31","2019-11-05","9-59M",state_dist.loc["kano"]),
                 ("kaduna","2019-11-16","2019-11-21","9-59M",state_dist.loc["kaduna"]),
                 ("zamfara","2019-11-16","2019-11-21","9-59M",state_dist.loc["zamfara"]),
                 ("jigawa","2019-11-16","2019-11-21","9-59M",state_dist.loc["jigawa"])] 

## 2019 outbreak response in Borno
## Comment: "First phase of ORI completed in 8 wards in Maiduguri LGA"
## Dates are from
## https://www.afro.who.int/news/who-and-partners-provide-life-saving-vaccine-more-12-million-children-against-measles-borno
sia_calendar += [("borno","2019-03-21","2019-03-25","6M-9Y",475778)]

## 2019 MCV2 introduction in the South
## Information about this is tentative. But for now, we use
## dates from https://www.bbc.com/pidgin/tori-50324143
## NB: I CANT FIND THE REACH FOR THIS ONE, ITS NOT IN THE SPREADSHEET
## Actually it went down to 9MO, see the slides from the CWG.
sia_calendar += [("imo","2019-11-14","2019-11-18","9-23M",np.nan),
                 ("anambra","2019-11-14","2019-11-18","9-23M",np.nan),
                 ("ebonyi","2019-11-14","2019-11-18","9-23M",np.nan),
                 ("abia","2019-11-14","2019-11-18","9-23M",np.nan),
                 ("enugu","2019-11-14","2019-11-18","9-23M",np.nan),
                 ("delta","2019-11-14","2019-11-18","9-23M",np.nan),
                 ("cross river","2019-11-14","2019-11-18","9-23M",np.nan),
                 ("bayelsa","2019-11-14","2019-11-18","9-23M",np.nan),
                 ("rivers","2019-11-14","2019-11-18","9-23M",np.nan),
                 ("akwa ibom","2019-11-14","2019-11-18","9-23M",np.nan),
                 ("edo","2019-11-14","2019-11-18","9-23M",np.nan),
                 ("lagos","2019-11-14","2019-11-18","9-23M",np.nan),
                 ("ekiti","2019-11-14","2019-11-18","9-23M",np.nan),
                 ("ondo","2019-11-14","2019-11-18","9-23M",np.nan),
                 ("osun","2019-11-14","2019-11-18","9-23M",np.nan),
                 ("oyo","2019-11-14","2019-11-18","9-23M",np.nan),
                 ("ogun","2019-11-14","2019-11-18","9-23M",np.nan)]

## 2020 follow up campaign
## Comment: (Kogi & Niger states)
total_reach = 2160253
state_dist = births.loc[["kogi","niger"],np.arange(2015,2020)].sum(axis=1)
state_dist = np.round(total_reach*(state_dist/(state_dist.sum()))).astype(int)
sia_calendar += [("niger","2020-10-9","2020-10-18","9-59M",state_dist.loc["niger"]),
                 ("kogi","2020-10-9","2020-10-18","9-59M",state_dist.loc["kogi"])]

## 2021 campaigns that have actually happened
## End dates are estimated from the 2019 campaign taking 5 to 6 days per
## state.
total_reach = 16203500
state_dist = births.loc[["abia","bayelsa","borno","ebonyi","imo","kaduna","kano",
                         "katsina","kebbi","kwara","taraba","yobe","sokoto"],
                         np.arange(2016,2021)].sum(axis=1)
state_dist = np.round(total_reach*(state_dist/(state_dist.sum()))).astype(int)
sia_calendar += [("abia","2021-11-24","2021-11-30","9-59M",state_dist.loc["abia"]),
                 ("bayelsa","2021-11-27","2021-12-1","9-59M",state_dist.loc["bayelsa"]),
                 ("borno","2021-11-20","2021-11-25","9-59M",state_dist.loc["borno"]),
                 ("ebonyi","2021-12-4","2021-12-10","9-59M",state_dist.loc["ebonyi"]),
                 ("imo","2021-11-24","2021-11-30","9-59M",state_dist.loc["imo"]),
                 ("kaduna","2021-11-18","2021-11-24","9-59M",state_dist.loc["kaduna"]),
                 ("kano","2021-11-21","2021-11-27","9-59M",state_dist.loc["kano"]),
                 ("katsina","2021-11-16","2021-11-21","9-59M",state_dist.loc["katsina"]),
                 ("kebbi","2021-11-27","2021-12-1","9-59M",state_dist.loc["kebbi"]),
                 ("kwara","2021-12-11","2021-12-16","9-59M",state_dist.loc["kwara"]),
                 ("taraba","2021-11-18","2021-11-24","9-59M",state_dist.loc["taraba"]),
                 ("yobe","2021-11-18","2021-11-24","9-59M",state_dist.loc["yobe"]),
                 ("sokoto","2021-11-27","2021-12-1","9-59M",state_dist.loc["sokoto"])]

## 2022 campaigns that have actually happened
## with, again, some missing reach data. These were pulled by
## Kurt from the country working group slides too.
## End dates are estimated from the 2019 campaign taking 5 to 6 days per
## state. Kogi was delayed to 2023 early. Reach number from the WHO
## spreadsheet
## Seems like reach data is only for the June states, based on the 
## start end columns there.
total_reach = 6281891
state_dist = births.loc[["ogun","gombe","lagos",
                         #"cross river","ondo",
                         #"bauchi","abuja","plateau","oyo","nasarawa",
                         #"anambra","osun","benue","enugu","ekiti","adamawa",
                         #"akwa ibom","niger","rivers","zamfara","edo","delta",
                         #"jigawa","kogi",
                         ],np.arange(2017,2022)].sum(axis=1)
state_dist = np.round(total_reach*(state_dist/(state_dist.sum())))#.astype(int)
sia_calendar += [("ogun","2022-05-26","2022-06-01","9-59M",state_dist.loc["ogun"]),
                 ("gombe","2022-06-12","2022-06-17","9-59M",state_dist.loc["gombe"]),
                 ("lagos","2022-06-12","2022-06-17","9-59M",state_dist.loc["lagos"]),
                 ("cross river","2022-11-07","2022-11-12","9-59M",np.nan),
                 ("ondo","2022-11-07","2022-11-12","9-59M",np.nan),
                 ("bauchi","2022-11-15","2022-11-20","9-59M",np.nan),
                 ("abuja","2022-11-15","2022-11-20","9-59M",np.nan),
                 ("plateau","2022-11-15","2022-11-20","9-59M",np.nan),
                 ("oyo","2022-11-17","2022-11-22","9-59M",np.nan),
                 ("nasarawa","2022-11-18","2022-11-23","9-59M",np.nan),
                 ("anambra","2022-11-20","2022-11-25","9-59M",np.nan),
                 ("osun","2022-11-21","2022-11-26","9-59M",np.nan),
                 ("benue","2022-11-23","2022-11-28","9-59M",np.nan),
                 ("enugu","2022-11-23","2022-11-28","9-59M",np.nan),
                 ("ekiti","2022-11-25","2022-12-01","9-59M",np.nan),
                 ("adamawa","2022-11-27","2022-12-03","9-59M",np.nan),
                 ("akwa ibom","2022-11-29","2022-12-04","9-59M",np.nan),
                 ("niger","2022-11-29","2022-12-04","9-59M",np.nan),
                 ("rivers","2022-11-29","2022-12-04","9-59M",np.nan),
                 ("zamfara","2022-11-30","2022-12-05","9-59M",np.nan),
                 ("edo","2022-12-03","2022-12-08","9-59M",np.nan),
                 ("delta","2022-12-05","2022-12-10","9-59M",np.nan),
                 ("jigawa","2022-12-06","2022-12-11","9-59M",np.nan),
                 ("kogi","2023-02-04","2023-02-09","9-59M",np.nan)]

## 2023 campaigns that happened
## with, again, some missing reach data. These were pulled by
## Kurt from the country working group slides too.
## End dates are estimated from the 2019 campaign taking 5 to 6 days per
## state. .
sia_calendar += [("kwara","2023-10-26","2023-10-31","9-59M",np.nan),
                 ("borno","2023-10-19","2023-10-24","9-59M",np.nan),
                 ("taraba","2023-11-25","2023-11-30","9-59M",np.nan),
                 ("yobe","2023-12-03","2023-12-08","9-59M",np.nan),
                 ("kaduna","2023-11-30","2023-12-04","9-59M",np.nan),
                 ("kano","2023-12-03","2023-12-08","9-59M",np.nan),
                 ("katsina","2023-10-25","2023-10-30","9-59M",np.nan),
                 ("kebbi","2023-11-18","2023-11-23","9-59M",np.nan),
                 ("sokoto","2023-11-02","2023-11-07","9-59M",np.nan),
                 ("abia","2023-11-22","2023-11-27","9-59M",np.nan),
                 ("ebonyi","2023-10-19","2023-10-24","9-59M",np.nan),
                 ("imo","2023-10-19","2023-10-24","9-59M",np.nan),
                 ("bayelsa","2023-12-02","2023-12-07","9-59M",np.nan)]

## Format the calendar as a dataframe
sia_calendar = pd.DataFrame(sia_calendar,
                            columns=["state","start_date","end_date",
                                     "age_group","doses"])
sia_calendar["start_date"] = pd.to_datetime(sia_calendar["start_date"])
sia_calendar["end_date"] = pd.to_datetime(sia_calendar["end_date"])

if __name__ == "__main__":

    ## Any need to interpolate the doses?
    full_avg_doses = sia_calendar.loc[(sia_calendar["start_date"].dt.year >= 2000),
                                 ["state","age_group","doses"]].copy()
    full_avg_doses = full_avg_doses.groupby(["state","age_group"]).mean()["doses"]

    ## And maybe interpolate with a restricted time window?
    avg_doses = sia_calendar.loc[(sia_calendar["start_date"].dt.year >= 2000),
                                 ["state","age_group","doses"]].copy()
    avg_doses = avg_doses.groupby(["state","age_group"]).mean()["doses"]
    avg_doses = avg_doses.reindex(full_avg_doses.index)#.fillna(full_avg_doses)

    ## Fill the 12-23M IRI effort with the discounted value from the
    ## 9 to 59M, with disount based on ward level tally data from 2022,
    ## as explained in the manuscript appendix.
    discount = 1 - 0.491158
    the_2019_campaign = [t for t in avg_doses.index if t[1] == "9-23M"]
    avg_doses.loc[the_2019_campaign] = np.round(discount*avg_doses.loc[
                                        [(t[0],"9-59M") for t in the_2019_campaign]
                                        ].values)
    
    ## Fill the nans in the calendar
    avgs = pd.Series(avg_doses.loc[zip(sia_calendar["state"],
                            sia_calendar["age_group"])].values,
                     index=sia_calendar.index)
    sia_calendar["doses"] = sia_calendar["doses"].fillna(avgs).astype(int)

    ## Remove campaigns that are too recent, with too little information
    sia_calendar = sia_calendar.loc[
                    sia_calendar["start_date"] <= "2023-10-01"]\
                    .reset_index(drop=True)
    
    ## So we have
    print("\nThe SIA calendar (with imputed doses)...")
    print(sia_calendar)
    sia_calendar.to_csv(os.path.join("_data",
                        "imputed_sia_calendar_by_state.csv"))