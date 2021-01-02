library(tidyverse)
library(lubridate)
library(viridis)
filter <- dplyr::filter

stations <- read_csv("current_bluebikes_stations.csv")

trips <- read_csv("202010-bluebikes-tripdata.csv") %>%
  rbind(read_csv("202011-bluebikes-tripdata.csv")) %>%
  mutate(id=seq_along(starttime))

stations <- trips %>% group_by(`start station id`) %>%
  slice(which.min(starttime)) %>%
  select(station_id=`start station id`,
         name=`start station name`,
         latitude=`start station latitude`,
         longitude=`start station longitude`)

arrivals <- select(trips, time=stoptime, station_id=`end station id`) %>%
  mutate(event='arrival')
departures <- select(trips, time=starttime, station_id=`start station id`) %>%
  mutate(event='departure')
events <- rbind(arrivals, departures)

times <- (events
  %>% mutate(wday=wday(time, week_start=1),
           hour=hour(time), total_weeks=n_distinct(week(time)))
  %>% group_by(wday, hour, station_id, event)
  %>% summarise(rate=n() / max(total_weeks))
  %>% spread(station_id, rate, fill=0.)
  %>% gather(station_id, rate, -wday,  -hour, -event)
  %>% spread(event, rate, fill=0.)
  %>% mutate(net=arrival - departure))

write_csv(times, "rates.csv")
write_csv(stations, "stations.csv")

(ggplot(dplyr::filter(times, station_id == 60),
        aes(wday, hour, fill=net, label=sprintf("%.1f", net)))
  + scale_y_continuous(trans='reverse')
  + geom_tile() + scale_fill_distiller(type='div') + geom_text(alpha=0.3))

# Incorporate rain? Smoothing?
