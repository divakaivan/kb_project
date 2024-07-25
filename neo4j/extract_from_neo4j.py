import pandas as pd
from neo4j import GraphDatabase

def create_driver(uri, user, password, timeout=300):
    return GraphDatabase.driver(uri, auth=(user, password), connection_timeout=timeout)

def load_data():
    uri = "neo4j://localhost:7687"
    driver = create_driver(uri, "neo4j", "password")
    with driver.session() as session:
        # Query for CreditCard nodes
        cc_query = """
        MATCH (c:CreditCard)
        RETURN c.cc_num AS cc_num, c.lat AS lat, c.long AS long
        """
        credit_card_nodes = pd.DataFrame([dict(record) for record in session.run(cc_query)])
        
        # Query for Merchant nodes
        merchant_query = """
        MATCH (m:Merchant)
        RETURN m.name AS merchant, m.merch_lat AS merch_lat, m.merch_long AS merch_long
        """
        merchant_nodes = pd.DataFrame([dict(record) for record in session.run(merchant_query)])
        
        # Query for Transaction edges
        transaction_query = """
        MATCH (c:CreditCard)-[t:TRANSACTION]->(m:Merchant)
        RETURN c.cc_num AS cc_num, m.name AS merchant, t.amt AS amt, t.category AS category, t.trans_date_trans_time AS trans_date_trans_time, t.is_fraud AS is_fraud
        """
        transactions = pd.DataFrame([dict(record) for record in session.run(transaction_query)])
        
    return credit_card_nodes, merchant_nodes, transactions

credit_card_nodes, merchant_nodes, transactions = load_data()