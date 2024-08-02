* **Docker** - used to containarise and self-host the below services
* **Mlflow** - used for easy model comparison during development. Can be improved by using cloud services (like GCP) for hosting, database and artifact store
* **Mage** - used for pipeline orchestration of the model training and real-time inference pipelines. Can be improved by hosting mage on the cloud
* **Neo4j** - used as a Graph database to store transaction data as nodes and edges. Can be improved by using Neo4j's AuraDB (hosted on the cloud)
* **Kafka** - used to ensure real-time transaction data processing
* **Grafana** - used for real-time dashboard creating and monitoring. Can be improved by using Grafana's cloud services
* **Streamlit** - used to host a model dictionary showing models, evaluation metrics, and feature importance graphs

###### Terraform can be used to provide Infrastructure as Code (IaC) for services that can be hosted on the cloud