# Load necessary packages
library(quantmod)
library(imputeTS)
library(forecast)
library(caret)
library(keras)
library(randomForest)
library(rpart)
library(lubridate)

# Fetch stock data
getSymbols("AAPL", src = "yahoo", from = Sys.Date() - 5*365, to = Sys.Date())
data <- na.omit(AAPL)

# Check for missing values
missing_values <- sum(is.na(data))

# Interpolate missing values if any
data <- na_interpolation(data)

# Check for outliers
# Using boxplot.stats requires converting xts object to numeric vector
outliers <- boxplot.stats(as.numeric(data$AAPL.Close))$out

# Plotting the data
plot(data$AAPL.Close, main = "AAPL Closing Price Over Time", ylab = "Price", xlab = "Date", col = "blue", type = "l")

# Splitting data
train_size <- floor(0.8 * nrow(data))
train <- data[1:train_size, ]
test <- data[(train_size + 1):nrow(data), ]

# Convert data to monthly
monthly_data <- to.monthly(data, indexAt = "lastof", OHLC = FALSE)
monthly_close <- monthly_data[, "AAPL.Close"]

# Convert to time series object
start_year <- year(index(monthly_close)[1])
start_month <- month(index(monthly_close)[1])
monthly_ts <- ts(monthly_close, frequency = 12, start = c(start_year, start_month))

# Decompose the time series
decomposition_add <- decompose(monthly_ts, type = "additive")
decomposition_mult <- decompose(monthly_ts, type = "multiplicative")

# Plot decomposition
plot(decomposition_add)
plot(decomposition_mult)

# Fit Holt-Winters model
hw_model <- HoltWinters(monthly_ts)
hw_forecast <- forecast(hw_model, h = 12)  # Forecast for the next year

# Plotting forecast
plot(hw_forecast)
lines(test$AAPL.Close, col = "red")

# ARIMA Model
arima_model <- auto.arima(monthly_ts)
checkresiduals(arima_model)

# Seasonal-ARIMA (SARIMA) Model
sarima_model <- auto.arima(monthly_ts, seasonal = TRUE)
checkresiduals(sarima_model)

# Forecast using ARIMA and SARIMA
forecast_arima <- forecast(arima_model, h = 3) # Forecast for next 3 months
forecast_sarima <- forecast(sarima_model, h = 3)

# Plot forecasts for ARIMA and SARIMA
plot(forecast_arima, main = "ARIMA Forecast")
plot(forecast_sarima, main = "SARIMA Forecast")

# Fit ARIMA to monthly series
arima_monthly <- auto.arima(monthly_ts)
forecast_arima_monthly <- forecast(arima_monthly, h = 12) # Forecast for the next year
plot(forecast_arima_monthly, main = "Monthly ARIMA Forecast")


# Prepare data for LSTM
scaled_data <- scale(data$AAPL.Close)
train_data <- scaled_data[1:train_size]
test_data <- scaled_data[(train_size + 1):nrow(data)]

# Check if there's enough data for training and testing
if(length(train_data) <= 60 | length(test_data) <= 60) {
  stop("Not enough data points to create LSTM input sequences.")
}

x_train <- array(train_data[1:(length(train_data) - 60)], dim = c(length(train_data) - 60, 60, 1))
y_train <- train_data[61:length(train_data)]

x_test <- array(test_data[1:(length(test_data) - 60)], dim = c(length(test_data) - 60, 60, 1))
y_test <- test_data[61:length(test_data)]

# Build LSTM model
model <- keras_model_sequential()
model %>%
  layer_lstm(units = 50, return_sequences = TRUE, input_shape = c(60, 1)) %>%
  layer_lstm(units = 50) %>%
  layer_dense(units = 1)

model %>% compile(
  optimizer = 'adam',
  loss = 'mean_squared_error'
)

# Fit the model
history <- model %>% fit(
  x_train, y_train,
  epochs = 25,
  batch_size = 32,
  validation_split = 0.2
)

# Forecasting
predicted_stock_price_scaled <- model %>% predict(x_test)

# Rescale predictions
predicted_stock_price <- predicted_stock_price_scaled * attr(scaled_data, 'scaled:scale') + attr(scaled_data, 'scaled:center')

# Plotting forecast
plot(data$AAPL.Close[(train_size + 1):nrow(data)], col = "red", type = "l", main = "LSTM Forecast", ylab = "Price", xlab = "Date")
lines(predicted_stock_price, col = "blue")


# Prepare data for tree-based models
data_lagged <- as.data.frame(embed(data, 2))
colnames(data_lagged) <- c("Lag1", "Current")

# Random Forest
rf_model <- randomForest(Current ~ Lag1, data = data_lagged)
rf_forecast <- predict(rf_model, newdata = data_lagged)

# Decision Tree
tree_model <- rpart(Current ~ Lag1, data = data_lagged)
tree_forecast <- predict(tree_model, newdata = data_lagged)

# Plot forecasts from Random Forest and Decision Tree
plot(data_lagged$Current, type = 'l', col = 'black', lty = 1, main = "Forecasts from Tree-based Models")
lines(rf_forecast, col = 'blue')
lines(tree_forecast, col = 'red')
legend("topright", legend = c("Actual", "Random Forest", "Decision Tree"), col = c("black", "blue", "red"), lty = 1)
