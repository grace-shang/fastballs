# Load libraries
library(car)
library(tidyverse)

# Loading and cleaning data
fb <- read_csv("savant_data.csv")
fb <- fb %>% select(player_id, total, `downward movement w/ gravity (in)`, 
                    `glove/arm-side movement (in)`, whiff_percent, `pitch (MPH)`,
                    `spin (RPM)`, `vertical release pt (ft)`, `extension (ft)`)

bot <- read_csv("botcmd.csv") %>% select(player_id, `botCmd FA`)

# Merging and cleaning data
fastballs <- merge(fb, bot, by = 'player_id')
renamed_cols <- c('downwards_mvmt' = 'downward movement w/ gravity (in)',
                  'side_mvmt' = 'glove/arm-side movement (in)',
                  'speed' = 'pitch (MPH)',
                  'spin' = 'spin (RPM)',
                  'release_pt' = 'vertical release pt (ft)',
                  'extension' = 'extension (ft)',
                  'command' = 'botCmd FA')

fastballs <- rename(fastballs, renamed_cols)
head(fastballs)

# ----------------------------------------------------------------------------#

# Splitting Data
set.seed(1000)
fastballs_training <- fastballs %>% sample_frac(size = 0.8)
fastballs_validation <- anti_join(fastballs, fastballs_training)
fastballs_training

# Model 1
reg_all <- lm(whiff_percent ~ speed + spin + downwards_mvmt + 
                side_mvmt + release_pt + extension + command, 
              data = fastballs_training)

summary(reg_all)

# Check Model 1 Assumptions
plot(reg_all)

plot(reg_all$residuals ~ fastballs_training$speed,
     main = 'Speed',
     xlab = 'Speed (MPH)', ylab = 'Residuals',
     col = 'lightcoral')
plot(reg_all$residuals ~ fastballs_training$spin,
     main = 'Spin',
     xlab = 'Spin (RPM)', ylab = 'Residuals',
     col = 'orange')
plot(reg_all$residuals ~ fastballs_training$downwards_mvmt,
     main = 'Downwards Movement',
     xlab = 'Downwards Movement (in)', ylab = 'Residuals',
     col = 'goldenrod')
plot(reg_all$residuals ~ fastballs_training$side_mvmt,
     main = 'Side Movement',
     xlab = 'Side Movement (in)', ylab = 'Residuals',
     col = 'darkseagreen')
plot(reg_all$residuals ~ fastballs_training$release_pt,
     main = 'Release Point',
     xlab = 'Release Point (ft)', ylab = 'Residuals',
     col = 'cadetblue')
plot(reg_all$residuals ~ fastballs_training$extension,
     main = 'Extension',
     xlab = 'Extension (ft)', ylab = 'Residuals',
     col = 'plum')
plot(reg_all$residuals ~ fastballs_training$command,
     main = 'Command',
     xlab = 'Command (Bot Command Rating)', ylab = 'Residuals',
     col = 'gray')

vif(reg_all)

# Addressing linearity in Model 1
MASS::boxcox(reg_all)

# Arcsine transformed response
asine_whiff_percent <- asin(sqrt(fastballs_training$whiff_percent / 100))

# Transformed regression
asine_reg_all <- lm(asine_whiff_percent ~ speed + spin + downwards_mvmt + 
                      side_mvmt + release_pt + extension + command, 
                    data = fastballs_training)
summary(asine_reg_all)

par(mfrow = c(1,2))
plot(asine_reg_all)
plot(reg_all) # Not as good as original

# Remove outlier (Bieber)
updated_fastballs_training <- fastballs_training %>%  filter(player_id != 669456) 

# Re-train Model 1 without transformation
reg_all_removed <- lm(whiff_percent ~ speed + spin + downwards_mvmt + 
                        side_mvmt + release_pt + extension + command, 
                      data = updated_fastballs_training)

summary(reg_all_removed)
par(mfrow = c(1,2))
plot(reg_all_removed)

# Re-train Arcsine version of Model 1
arcsine_reg_all_removed <- lm(asin(sqrt(updated_fastballs_training$whiff_percent / 100)) ~
                                speed + spin + downwards_mvmt + side_mvmt + release_pt + 
                                extension + command, data = updated_fastballs_training)
summary(arcsine_reg_all_removed)

par(mfrow = c(2,2))
plot(arcsine_reg_all_removed) # Not as good as original


# Model 2
reg_sdsr <- lm(whiff_percent ~ spin + downwards_mvmt + side_mvmt + 
                 release_pt, data = updated_fastballs_training)

summary(reg_sdsr)

# Anova test for Model 2
anova(reg_all_removed, reg_sdsr) # Not significant

# Check assumptions for Model 2
plot(reg_sdsr) # Some heteroskedasticity

plot(reg_sdsr$residuals ~ updated_fastballs_training$spin,
     main = 'Spin',
     xlab = 'Spin (RPM)', ylab = 'Residuals',
     col = 'orange')
plot(reg_sdsr$residuals ~ updated_fastballs_training$downwards_mvmt,
     main = 'Downwards Movement',
     xlab = 'Downwards Movement (in)', ylab = 'Residuals',
     col = 'goldenrod')
plot(reg_sdsr$residuals ~ updated_fastballs_training$side_mvmt,
     main = 'Side Movement',
     xlab = 'Side Movement (in)', ylab = 'Residuals',
     col = 'darkseagreen')
plot(reg_sdsr$residuals ~ updated_fastballs_training$release_pt,
     main = 'Release Point',
     xlab = 'Release Point (ft)', ylab = 'Residuals',
     col = 'cadetblue')

# Attempt to correct heteroskedasticity using arcsine transformation
arcsine_reg_sdsr_removed <- lm(asin(sqrt(updated_fastballs_training$whiff_percent / 100)) ~
                                 spin + downwards_mvmt + side_mvmt + release_pt, 
                               data = updated_fastballs_training)

summary(arcsine_reg_sdsr_removed) # Doesn't help

# Re-check
plot(arcsine_reg_sdsr_removed)
plot(arcsine_reg_sdsr_removed$residuals ~ updated_fastballs_training$spin,
     main = 'Spin',
     xlab = 'Spin (RPM)', ylab = 'Residuals',
     col = 'orange')
plot(arcsine_reg_sdsr_removed$residuals ~ updated_fastballs_training$downwards_mvmt,
     main = 'Downwards Movement',
     xlab = 'Downwards Movement (in)', ylab = 'Residuals',
     col = 'goldenrod')
plot(arcsine_reg_sdsr_removed$residuals ~ updated_fastballs_training$side_mvmt,
     main = 'Side Movement',
     xlab = 'Side Movement (in)', ylab = 'Residuals',
     col = 'darkseagreen')
plot(arcsine_reg_sdsr_removed$residuals ~ updated_fastballs_training$release_pt,
     main = 'Release Point',
     xlab = 'Release Point (ft)', ylab = 'Residuals',
     col = 'cadetblue')

# Attempt to correct heteroskedasticity using Weighted Least Squares
weights <- 1 / lm(abs(reg_sdsr$residuals) ~ reg_sdsr$fitted.values)$fitted.values^2

wls_reg_sdsr <- lm(whiff_percent ~ spin + downwards_mvmt + side_mvmt + release_pt,
                   data = updated_fastballs_training, weights = weights)

summary(wls_reg_sdsr)

# Re-check
par(mfrow = c(1,2))
plot(wls_reg_sdsr)

plot(wls_reg_sdsr$residuals ~ updated_fastballs_training$spin,
     main = 'Spin',
     xlab = 'Spin (RPM)', ylab = 'Residuals',
     col = 'orange')
plot(wls_reg_sdsr$residuals ~ updated_fastballs_training$downwards_mvmt,
     main = 'Downwards Movement',
     xlab = 'Downwards Movement (in)', ylab = 'Residuals',
     col = 'goldenrod')
plot(wls_reg_sdsr$residuals ~ updated_fastballs_training$side_mvmt,
     main = 'Side Movement',
     xlab = 'Side Movement (in)', ylab = 'Residuals',
     col = 'darkseagreen')
plot(wls_reg_sdsr$residuals ~ updated_fastballs_training$release_pt,
     main = 'Release Point',
     xlab = 'Release Point (ft)', ylab = 'Residuals',
     col = 'cadetblue')

# Check multicollinearity
vif(wls_reg_sdsr) # None significant

# Attempt to simplify further
reg_dr <- lm(whiff_percent ~ downwards_mvmt + release_pt,
             data = updated_fastballs_training)
summary(reg_dr)

# Partial F test
anova(wls_reg_sdsr, reg_dr)

# Try removing one at a time
# Remove side movement
reg_sdr <- lm(whiff_percent ~ spin + downwards_mvmt + release_pt,
              data = updated_fastballs_training)
summary(reg_sdr)

anova(wls_reg_sdsr, reg_sdr) # Significant

# Remove spin
reg_dsr <- lm(whiff_percent ~ downwards_mvmt + side_mvmt + release_pt,
              data = updated_fastballs_training)
summary(reg_dsr)
anova(wls_reg_sdsr, reg_sdr) # Significant


# --------------------------------------------------------------------------- #

# Model Selection
models = c('All', 'Speed + Downwards Movement + Sideways Movement + Release Point')

n_predictors = c(7, 4)

adj_r_squared = c(summary(reg_all_removed)$adj.r.squared,
                  summary(wls_reg_sdsr)$adj.r.squared)

bic = c(BIC(reg_all_removed), 
        BIC(wls_reg_sdsr))

aic_corrected = c(AIC(reg_all_removed) + (2*9*10)/(119-9+1), 
                  AIC(wls_reg_sdsr) + (2*6*7)/(103-6+1))

selection <- tibble(models, n_predictors, adj_r_squared, aic_corrected, bic)
selection

# --------------------------------------------------------------------------- #

# Model Validation
reg_sdsr_validation <- lm(whiff_percent ~ spin + downwards_mvmt + side_mvmt + release_pt,
                          data = fastballs_validation)

weights_val <- 1 / lm(abs(reg_sdsr_validation$residuals) ~ reg_sdsr_validation$fitted.values)$fitted.values^2

wls_reg_sdsr_val <- lm(whiff_percent ~ spin + downwards_mvmt + side_mvmt + release_pt,
                       data = fastballs_validation, weights = weights_val)

summary(wls_reg_sdsr_val)

# Columns
models = c('Training', 'Validation')
n = c(119, 30)
adj_r_squared = c(summary(wls_reg_sdsr)$adj.r.squared,
                  summary(wls_reg_sdsr_val)$adj.r.squared)
intercept_estimate <- c(wls_reg_sdsr$coefficients[1],
                        wls_reg_sdsr_val$coefficients[1])
spin_estimate <- c(wls_reg_sdsr$coefficients[2],
                   wls_reg_sdsr_val$coefficients[2])
downwards_mvmt_estimate <- c(wls_reg_sdsr$coefficients[3],
                             wls_reg_sdsr_val$coefficients[3])
side_mvmt_estimate <- c(wls_reg_sdsr$coefficients[4],
                        wls_reg_sdsr_val$coefficients[4])
release_pt_estimate <- c(wls_reg_sdsr$coefficients[5],
                         wls_reg_sdsr_val$coefficients[5])

# Validation table
validation <- tibble(models, n, adj_r_squared, spin_estimate, downwards_mvmt_estimate,
                     side_mvmt_estimate, release_pt_estimate)
validation

# Check assumptions
plot(wls_reg_sdsr_val) # Leverage point
vif(wls_reg_sdsr_val)

# Compare datasets response variables
# Testing
summarize(updated_fastballs_training, pitchers = n(), 
          `min whiff` = min(whiff_percent), `max whiff` = max(whiff_percent),
          `average whiff` = mean(whiff_percent), `median whiff` = median(whiff_percent), 
          `sd whiff` = sd(whiff_percent))

# Validation
summarize(fastballs_validation, pitchers = n(), 
          `min whiff` = min(whiff_percent), `max whiff` = max(whiff_percent),
          `average whiff` = mean(whiff_percent), `median whiff` = median(whiff_percent), 
          `sd whiff` = sd(whiff_percent))

# Check average number of fastballs thrown
mean(updated_fastballs_training$total)
mean(fastballs_validation$total)