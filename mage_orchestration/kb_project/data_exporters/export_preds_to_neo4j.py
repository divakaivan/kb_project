from neo4j import GraphDatabase
from mage_ai.streaming.sinks.base_python import BasePythonSink
from typing import Callable, Dict, List

if 'streaming_sink' not in globals():
    from mage_ai.data_preparation.decorators import streaming_sink

def create_driver(uri, user, password, timeout=300):
    return GraphDatabase.driver(uri, auth=(user, password), connection_timeout=timeout)

def insert_transaction(tx, row):
    result = tx.run("""
        MATCH (c:CreditCard {cc_num: $cc_num})
        MATCH (m:Merchant {name: $merchant})
        OPTIONAL MATCH (c)-[t:TRANSACTION {trans_num: $trans_num}]->(m)
        WITH c, m, t
        WHERE t IS NULL
        MERGE (c)-[:TRANSACTION {
            trans_date_trans_time: $trans_date_trans_time, 
            amt: $amt, 
            category: $category, 
            trans_num: $trans_num, 
            is_fraud: $is_fraud,
            pred_gcn_is_fraud: $pred_gcn_is_fraud,
            pred_xgb_is_fraud: $pred_xgb_is_fraud,
            pred_catboost_is_fraud: $pred_catboost_is_fraud
        }]->(m)
        RETURN t
    """, row)
    return result.single()

@streaming_sink
class CustomSink(BasePythonSink):

    def batch_write(self, messages: List[Dict]):

        uri = "bolt://neo4j:7687"
        driver = create_driver(uri, "neo4j", "password")

        for msg in messages:
            with driver.session() as session:
                session.write_transaction(insert_transaction, msg)
