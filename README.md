# MachineLearning_MusicDataset


Proyecto de Machine Learning sin Framework. Se busca analizar y clasificar información recabada sobre usuarios de una aplicación de música por streaming, esto con el propósito de entender y predecir a los usuarios que cancelan su sucripción.

Se implementa una regresión logística sin uso de un framework enfocado en Machine Learning.

Información de dataset
41,700 muestras 
19 atributos (incluyendo target)

Columnas:
* customer_id - a unique customer identification number
* age - the age of the user
* location - the US state of the user
* subscription_type - type of subsciption
* payment_plan - how often the user pays, monthly of annually
* num_subscription_pauses - number of times the user has paused their subscription (max 2)
* payment_method - form of user payment
* customer_service_inquiries - the frequency of customer service inquiries from the user
* signup_date - date the user signed up for the music subscription service
* weekly_hours - average number of weekly listening hours
* average_session_length - average length of each music listening session (in hours)
* song_skip_rate - percentage of songs the user does not finish
* weekly_songs_played - average number of songs the user plays in a week
* weekly_unique_songs - average number of unique songs the user plays in a week
* num_favorite_artists - number of artists the user set as favorite artists
* num_platform_friends - number of user connections in the app
* num_playlists_created - number of playlists the user created
* num_shared_playlists - number of playlists that are shared publicly
* notifications_clicked - number of in-app notifications clicked on
* churned - this is the target variable, 0 = customer is active, 1 = customer churned
