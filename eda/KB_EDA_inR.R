library(dplyr)
data <- read.csv("/Users/choijaehyeok/Desktop/KB_data/fraudTrain.csv")
data %>% head()

#사용 열 추출
use_data <- data[, 12:ncol(data)]
use_data %>% head()

#데이터형식 파악
use_data %>% sapply(class)

#결측치확인
use_data %>% is.na() %>% sum()


#job종류
use_data%>%
  count(job) %>%
  arrange(desc(n)) %>%
  head()

#city_pop
unique(use_data$state)

use_data %>% select(state, city_pop) %>% 
  group_by(state) %>% count(city_pop) %>% arrange(n)


#is_fraud
fraud <- use_data %>% select(state,merch_lat,merch_long,is_fraud) %>% 
  filter(is_fraud == 1)


library(ggmap)
register_google(key = 'AIzaSyBbHMrm3RbTqqSJzQePf1J-XvLpY30XOnE')

# 미국 지도 얻기
usa_map <- get_googlemap(center = c(lon = -95.7129, lat = 37.0902), zoom = 4, maptype = "terrain")

# 데이터 시각화
ggmap(usa_map) +
  geom_point(data = fraud, aes(x = merch_long, y = merch_lat, color = state), size = 2, alpha = 0.2) +
  labs(title = "Merchant Locations by State", 
       x = "Longitude", 
       y = "Latitude") +
  theme_minimal()

#주별로 어디가 가장 사기를 많이 당했는지 - NY(New York)
fraud_count <- fraud %>% group_by(state) %>% count()

# 막대 그래프 그리기
ggplot(fraud_count, aes(x = reorder(state, -n), y = n)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(title = "Number of Cases by State", 
       x = "State", 
       y = "Number of Cases") +
  theme_minimal() +
  theme_light()+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))



#dob별로 사기 건수

# 연도 추출 및 연도대 생성
use_data <- use_data %>%
  mutate(year = as.numeric(format(as.Date(dob, format="%Y-%m-%d"), "%Y")),
         decade = floor(year / 10) * 10)

# 연도대별 사기 건수 계산
fraud_by_decade <- use_data %>%
  filter(is_fraud == 1) %>%
  group_by(decade) %>%
  summarise(fraud_count = n())

fraud_by_decade


# 결과 시각화
ggplot(fraud_by_decade, aes(x = factor(decade), y = fraud_count)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(title = "Fraud Cases by Decade of Birth",
       x = "Decade of Birth",
       y = "Number of Fraud Cases") +
  theme_minimal()




###주별 인구수와 그 인구수에 따른 사기비율
# 주별 총 인구수 계산
state_population <- use_data %>%
  group_by(state) %>%
  summarise(total_population = sum(city_pop))

# 주별 사기 건수 및 사기율 계산
state_fraud <- use_data %>%
  group_by(state) %>%
  summarise(fraud_count = sum(is_fraud == 1),
            total_transactions = n()) %>%
  left_join(state_population, by = "state") %>%
  mutate(fraud_rate = fraud_count / total_transactions * 100,
         fraud_per_capita = fraud_count / total_population * 100000)
# 사기율 시각화
ggplot(state_fraud, aes(x = reorder(state, -fraud_rate), y = fraud_rate)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(title = "Fraud Rate by State",
       x = "State",
       y = "Fraud Rate (%)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# 인구당 사기 건수 시각화
ggplot(state_fraud, aes(x = reorder(state, -fraud_per_capita), y = fraud_per_capita)) +
  geom_bar(stat = "identity", fill = "darkorange") +
  labs(title = "Fraud Cases per 100,000 People by State",
       x = "State",
       y = "Fraud Cases per 100,000 People") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))



###Unix_time
library(lubridate)

# unix_time을 날짜 및 시간 형식으로 변환
use_data <- use_data %>%
  mutate(datetime = as_datetime(unix_time),
         hour = hour(datetime))

# 시간대별 사기 건수 계산
fraud_by_hour <- use_data %>%
  filter(is_fraud == 1) %>%
  group_by(hour) %>%
  summarise(fraud_count = n())

# 결과 시각화
ggplot(fraud_by_hour, aes(x = hour, y = fraud_count)) +
  geom_line(color = "steelblue", size = 1) +
  geom_point(color = "steelblue", size = 2) +
  labs(title = "Number of Fraud Cases by Hour of the Day",
       x = "Hour of the Day",
       y = "Number of Fraud Cases") +
  scale_x_continuous(breaks = 0:23) +
  theme_minimal()
