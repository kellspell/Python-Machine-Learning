from  statistics import mode
import scipy.stats as stats

data = [12,21,34,54,67,89]
mean = sum(data) / len(data)
print("Mean:", mean)

sorted_data = sorted(data)
median = sorted_data[len(data) // 2] if len(data) % 2 != 0 else \
    (sorted_data[len(data) // 2 - 1] + sorted_data[len(data) // 2]) /2
print("Median:", median)  

print("Mode:", mode(data))    

variance = sum((x - mean) ** 2 for x in data) / len(data)
print("Varience: ", variance)

std_dev = variance ** 0.5 
print("Standaded deb", std_dev) 

simple_mean = mean
z_core = 1.96

ci = (simple_mean - z_core * (std_dev / len(data) ** 0.5),
      simple_mean + z_core * (std_dev / len(data) ** 0.5))
print("CI:", ci)