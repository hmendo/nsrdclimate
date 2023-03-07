






# =============================================================================
# 
# =============================================================================


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 10:43:38 2023

@author: hmendo
"""

library(ggplot2)

load("audrey_sandbox/02_array_info.Rdata")
load("audrey_sandbox/01b_model_data_dfs.Rdata")
load("audrey_sandbox/00_model_names.Rdata")
load("audrey_sandbox/00_site_df.Rdata")

precip_files_all <- list.files("\\\\snl\\Collaborative\\nsrd_climate_impacts\\data\\raw\\precipitation\\")

mymodel <- models[sample(1:length(models),1)]
myfile <- precip_files_all[grep(mymodel, precip_files_all)[sample(1:14,1)]]
mysite <- sites[sample(1:nrow(sites), 1),"site_name"]

nc_df <- model_dfs[[mymodel]]
data_df <- array_info[[mymodel]]

data_lon <- data_df$data_lon[which(data_df$site_name == mysite)]
data_lat <- data_df$data_lat[which(data_df$site_name == mysite)]

plot_lon <- nc_df[which(round(nc_df$lon) == round(data_lon)),]
plot_lat <- nc_df[which(round(nc_df$lat) == round(data_lat)),]
plot_nc <- rbind(plot_lon, plot_lat)
plot_nc <- plot_nc[!duplicated(plot_nc),]
plot_nc <- subset(plot_nc, lon > round(data_lon)-1.5 & lon < round(data_lon)+1.5)
plot_nc <- subset(plot_nc, lat > round(data_lat)-1.5 & lat < round(data_lat)+1.5)
rm(plot_lat, plot_lon)

# which(plot_nc$lon == data_lon)
# plot_nc[110,] == c(data_lon, data_lat)
close_point <- plot_nc[which(plot_nc$lon == data_lon),]
site_loc <- data.frame(lon = data_df$site_lon[which(data_df$site_name == mysite)],
                       lat = data_df$site_lat[which(data_df$site_name == mysite)])

ggplot(data = NULL, aes(x = lon, y = lat)) +
  geom_point(data = plot_nc) +
  geom_point(data = close_point, color = "red", size = 3, shape = 3) +
  geom_point(data = site_loc, color = "#008080", size = 4) +
  ggtitle(paste(mysite, mymodel)) +
  theme_bw()