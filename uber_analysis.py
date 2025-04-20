# Step 1: Import Libraries  
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  

# Step 2: Load Data  
df = pd.read_csv("uber-raw-data-apr14.csv")  

# Step 3: Data Cleaning  
# Check missing values  
print("Missing values:\n", df.isnull().sum())  

# Handle missing values (drop or fill)  
df = df.dropna()  # Drop rows with missing values  

# Convert 'Date/Time' to datetime  
df['Date/Time'] = pd.to_datetime(df['Date/Time'])  

# Extract features (hour, day, weekday)  
df['Hour'] = df['Date/Time'].dt.hour  
df['Day'] = df['Date/Time'].dt.day  
df['Weekday'] = df['Date/Time'].dt.weekday  # 0=Monday, 6=Sunday  

print("Data cleaned successfully!")  
# Step 4: Analyze Peak Ride Times  
# Count rides by hour  
hourly_rides = df['Hour'].value_counts().sort_index()  

# Plot peak hours  
plt.figure(figsize=(12, 6))  
plt.plot(hourly_rides.index, hourly_rides.values, marker='o', color='blue')  
plt.title("Peak Ride Times (April 2014)")  
plt.xlabel("Hour of Day")  
plt.ylabel("Number of Rides")  
plt.axvspan(17, 19, color='red', alpha=0.2, label='Peak Hours (5 PM - 7 PM)')  
plt.legend()  
plt.savefig("peak_hours.png")  # Save plot  
plt.show()  

# Step 5: Popular Ride Purposes (if available)  
# If dataset has a 'Purpose' column:  
if 'Purpose' in df.columns:  
    purpose_counts = df['Purpose'].value_counts()  
    plt.figure(figsize=(10, 6))  
    purpose_counts.plot(kind='bar', color='green')  
    plt.title("Popular Ride Purposes")  
    plt.xlabel("Purpose")  
    plt.ylabel("Count")  
    plt.xticks(rotation=45)  
    plt.savefig("ride_purposes.png")  
    plt.show()  

# Step 6: Geographic Hotspots (Manhattan)  
# Filter coordinates for Manhattan  
manhattan_df = df[(df['Lat'] >= 40.7) & (df['Lat'] <= 40.8) &  
                  (df['Lon'] >= -74.02) & (df['Lon'] <= -73.93)]  

print(f"Rides in Manhattan: {len(manhattan_df)} ({len(manhattan_df)/len(df)*100:.1f}%)")  
# Step 7: Correlation Analysis  
# Convert weekday to numerical (optional)  
df['Weekday'] = df['Date/Time'].dt.weekday  

# Create correlation matrix (example)  
correlation = df[['Hour', 'Weekday', 'Lat', 'Lon']].corr()  

# Plot heatmap  
plt.figure(figsize=(10, 6))  
plt.imshow(correlation, cmap='coolwarm', interpolation='none')  
plt.colorbar()  
plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=45)  
plt.yticks(range(len(correlation.columns)), correlation.columns)  
plt.title("Feature Correlation Heatmap")  
plt.savefig("correlation_heatmap.png")  
plt.show()  