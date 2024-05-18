## Packages
library(data.table)
library(fields)
library(gstat)
library(ggplot2)
library(gridExtra)
library(car)
library(MASS)
library(PerformanceAnalytics)
library(EnvStats)
library(stats)
library(automap)
library(sp)
library(sf)

#Import data
data = read.csv("C:/Users/mares/[COURS]/[MEMOIRE]/Mon_Code/[PetiteSeumoy]/rotated_test_Seumoy.csv")

# Create a theme for centered titles
centered_title <- theme(
  plot.title = element_text(hjust = 0.5)
)
leg.pos <- theme(legend.position = c(0.75, 0.15))
scale_x <- scale_x_continuous(limits = c(0, 10)) 
scale_y <-  scale_y_continuous(limits = c(0, 1700))

# 
head(data)
summary(data)
hist(data$z, main="Histogram of GPR Values", xlab="GPR Values", ylab="Frequency")

# Selection de la zone d'étude :
data <- subset(data, xr >= 627870 & xr <= 628270)
data <- subset(data, yr >= 5598980 & yr <= 5599200)

# Create a plot for Soil Moisture
ggplot() +
  geom_tile(data = data, aes(x = xr, y = yr, fill = z), width = 0.01, height = 0.01) +
  geom_point(data = data, aes(x = xr, y = yr), shape = 16, size = 0.5, color = "black") +
  scale_fill_gradientn(name = "Humidité du sol", colors = c('#87CEEB', '#4682B4', '#000080')) +
  theme(legend.key = element_rect(fill = "transparent", color = NA)) +
  xlab("Longitude") + ylab("Latitude") +
  ggtitle("Carte de l'Humidité du Sol") +
  coord_fixed(ratio = 1) +
  scale_x_continuous(breaks = pretty(data$xr, n = 10)) +
  scale_y_continuous(breaks = pretty(data$yr, n = 15))


##  Krigeage : 
# Build a variogram 
g <- gstat(id = "z", formula = z ~ 1, data = data, locations = ~xr + yr)
z.vario <- variogram(g)
print(z.vario);plot(z.vario)
z.vario.model <- vgm(psill=5e-05, model='Sph', range = 19, nugget = 1e-4)
plot(z.vario,z.vario.model)
z.vario.model.fit <- fit.variogram(z.vario,z.vario.model)
plot(z.vario,z.vario.model.fit)
print(z.vario.model.fit)
   
# Create a prediction grid 
gridsize = 4
xr_values <- seq(min(data$xr), max(data$xr), by = gridsize)
yr_values <- seq(min(data$yr), max(data$yr), by = gridsize)
data.grid <- expand.grid(xr = xr_values, yr = yr_values)

# Kriging for Calcium
z.krig <- krige(formula = z ~ 1, data = data, locations = ~xr + yr, newdata = data.grid, model = z.vario.model.fit, maxdist = 20)
# Create a plot for Calcium kriging prediction

ggplot() + 
  geom_tile(data = z.krig, aes(x = xr, y = yr, fill = var1.pred)) +
  geom_point(data = na.omit(data), aes(x = xr, y = yr), shape = 16, size = 0.01, color = "black") +
  scale_fill_gradientn(name = "Humidité du sol", colors = c('#87CEEB', '#4682B4', '#000080')) +
  theme(legend.key = element_rect(fill = "transparent", color = NA)) +
  xlab("X") + ylab("Y") + centered_title

# Create a plot for Calcium kriging prediction
ggplot() + 
  geom_tile(data = z.krig, aes(x = xr, y = yr, fill = var1.pred)) +
  scale_fill_gradientn(name = "Volumetric soil moisture (%)", colors = c('#87CEEB', '#4682B4', '#000080')) +
  theme(legend.key = element_rect(fill = "transparent", color = NA)) +
  coord_fixed(ratio = 1) +  # Cette ligne fixe le ratio des axes à 1
  xlab("longitude") + ylab("latitude") + ggtitle("Krigeage of volumetric soil moisture") + centered_title


# Exporter le data frame vers un fichier CSV
data_export <- as.data.frame(z.krig)  # Utiliser les résultats du krigeage
colnames(data_export)[colnames(data_export) == "var1.pred"] <- "z"
write.csv(data_export, "C:/Users/mares/[COURS]/[MEMOIRE]/Mon_Code/krigeage4X4_Seumoy.csv", row.names = FALSE)



