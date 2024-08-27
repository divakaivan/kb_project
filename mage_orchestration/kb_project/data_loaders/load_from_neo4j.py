import pandas as pd
from neo4j import GraphDatabase

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

def create_driver(uri, user, password, timeout=300):
    return GraphDatabase.driver(uri, auth=(user, password), connection_timeout=timeout)

@data_loader
def load_data(*args, **kwargs):
    uri = "bolt://neo4j:7687"
    driver = create_driver(uri, "neo4j", "password")
    with driver.session() as session:
        # Query for CreditCard nodes
        cc_query = """
        MATCH (c:CreditCard)
        RETURN c.cc_num as cc_num, c.city as city, c.state as state, c.job as job, c.lat as lat, c.long as long
        """
        credit_card_nodes = pd.DataFrame([dict(record) for record in session.run(cc_query)])
        # Query for Merchant nodes
        merchant_query = """
        MATCH (m:Merchant)
        RETURN m.name AS merchant, m.merch_lat as merch_lat, m.merch_long as merch_long
        """
        merchant_nodes = pd.DataFrame([dict(record) for record in session.run(merchant_query)])
        
        # Query for Transaction edges
        transaction_query = """
        MATCH (c:CreditCard)-[t:TRANSACTION]->(m:Merchant)
        RETURN c.cc_num AS cc_num, m.name AS merchant, t.amt AS amt, t.category AS category, t.trans_date_trans_time AS trans_date_trans_time, t.is_fraud AS is_fraud
        """
        transactions = pd.DataFrame([dict(record) for record in session.run(transaction_query)])
        
    return credit_card_nodes, merchant_nodes, transactions

@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
