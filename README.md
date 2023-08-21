# Weather Data Analysis and LSTM Model Development for Soil Moisture Prediction

### Project Overview

The purpose of this project is to analyze weather data from multiple sources, identify significant correlations with soil moisture data, and subsequently develop an LSTM model to predict soil moisture levels accurately. In addition, a simple app or webpage will be built to display real-time soil moisture percentages.
Data Sources and Collection Periods
Timespan of all three datasets: 30.1.2023. – 30.7.2023.

Our dataset includes weather data from two distinct sources and soil moisture data:
1. Davis Vue Weather Station: Weather data has been collected every 5 minutes from January 30, 2023, to June 15, 2023, and every 15 minutes thereafter until July 31, 2023.
2. Accuweather: Data was collected randomly on an hourly basis via an API. From 30.1.2023 until 24.6.2023.
3. Soil Moisture Data: Data was initially collected every 30 seconds from January 30, 2023, but switched to a frequency of every 10 minutes starting from May 25, 2023.

Please note there are two temporal gaps in the data due to sensor malfunctions:
• The first from February 15, 2023, at 10:26 until February 20, 2023, at 17:30.
• The second from May 19, 2023, at 18:11 until May 25, 2023, at 16:30.

Data is collected from two sensors, from graph it is obvious that sensor2 gave better results.

### Key Tasks
1. Comparative Data Analysis: Analyze weather data from both Accuweather and our local weather station. The following measurements from both sources will be compared: Temperature (°C), Relative Humidity (%), Dew Point (°C), Wind Speed (km/h), Wind Gust Speed (km/h), UV Index, Wet Bulb (°C), and Pressure (mb). This task aims to determine the reliability of Accuweather data for future use by comparing its accuracy with data from our own station. The acceptable margin of difference is up to 5%.
2. Correlation Study: Identify and quantify the relationships between various weathe r measurements and soil moisture levels. This study aims to understand the degree to which changes in weather conditions can predict variations in soil moisture, providing the foundation for developing an effective predictive model. To do this, you will have to scope soil moisture data in the range of 0-1 so it is not left out in Ohm values. Real situation on field has reached it extremes, minimum and maximum, so that is also recorded by data.
3. LSTM Model Development: Use the findings from the correlation study to develop a Long Short- Term Memory (LSTM) machine learning model that can predict soil moisture percentage based on weather data. The model should achieve an accuracy of at least 90% during development testing.
4. Real-time Testing: After the LSTM model is developed and achieves the desired accuracy, the model will be connected to our weather and soil moisture data servers for real-time testing. This phase will evaluate the model's performance against new, live data that it was neither trained on nor tested against during development.
5. App/Webpage Development: Develop a simple app or webpage that displays real-time soil moisture percentages, as predicted by the LSTM model. This will serve as a user-friendly interface for monitoring soil moisture levels in real-time.
Timeline and Milestones

### The project should be completed and ready for testing by September 1, 2023. The project is divided into three key milestones:
1. Completion of the comparative data analysis.
2. Successful development of the LSTM model with an accuracy of ≥90%.
3. Successful real-time testing of the model and completion of the app/webpage development.

### Key Challenge

One significant challenge in this project is the alignment of timestamps from the three different datasets. The frequencies of data collection varied across the datasets, necessitating careful synchronization based on timestamps for accurate data analysis and modeling.
Preferred Tools

Python is suggested as the language of choice for this project, due to its extensive use in similar projects and the wide range of scientific libraries available, such as pandas for data manipulation, NumPy for numerical computation, and TensorFlow or PyTorch for the development of the LSTM model. But also, you can recommend your preferred tool or software used to complete these tasks.

_Upon agreement on this job proposal, we will need to address data security and intellectual property rights agreements to ensure the safe handling of our data and secure the rights to the developed model and work. We're here to answer any questions or clarify any points within the scope of this project. We look forward to your expert input and engagement._
