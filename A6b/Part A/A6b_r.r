# Load necessary libraries
library(quantmod)
library(rugarch)
library(ggplot2)

# Download Google stock data
getSymbols("GOOGL", from = "2023-01-01", to = "2023-12-31")
google_data <- na.omit(GOOGL[, "GOOGL.Adjusted"])
google_returns <- diff(log(google_data))[-1]  # Calculate log returns and remove NA

# Plot the adjusted closing prices
ggplot(data = as.data.frame(google_data), aes(x = index(google_data), y = GOOGL.Adjusted)) +
  geom_line(color = 'blue') +
  labs(title = "Google (Alphabet Inc.) Adjusted Closing Prices", x = "Date", y = "Adjusted Close") +
  theme_minimal()

# Plot the log returns
ggplot(data = as.data.frame(google_returns), aes(x = index(google_returns), y = google_returns)) +
  geom_line(color = 'red') +
  labs(title = "Google (Alphabet Inc.) Log Returns", x = "Date", y = "Log Returns") +
  theme_minimal()

# Specify and fit GARCH(1,1) model
spec <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
                   mean.model = list(armaOrder = c(0, 0)))
garch_fit <- ugarchfit(spec = spec, data = google_returns)
print(garch_fit)

# Forecast the next 90 days of volatility
garch_forecast <- ugarchforecast(garch_fit, n.ahead = 90)
sigma_forecast <- sigma(garch_forecast)

# Create a data frame for plotting the forecasted volatility
forecast_dates <- seq.Date(from = as.Date("2024-01-01"), by = "day", length.out = 90)
forecast_data <- data.frame(Date = forecast_dates, Forecasted_Volatility = as.numeric(sigma_forecast))

# Plot the forecasted volatility
ggplot(forecast_data, aes(x = Date, y = Forecasted_Volatility)) +
  geom_line(color = 'green') +
  labs(title = "Forecasted Volatility for Google (Alphabet Inc.)", x = "Date", y = "Forecasted Volatility") +
  theme_minimal()
