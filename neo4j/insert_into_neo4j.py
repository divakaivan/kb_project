from neo4j import GraphDatabase
import pandas as pd
import time

# change to your csv name
df = pd.read_csv('name.csv', index_col=0)

# Basic data preprocessing
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], format='%Y-%m-%d %H:%M:%S')
df['dob'] = pd.to_datetime(df['dob'], format='%Y-%m-%d')
df['merchant'] = df['merchant'].str.replace('fraud_', '')

def create_driver(uri, user, password, timeout=300):
    return GraphDatabase.driver(uri, auth=(user, password), connection_timeout=timeout)

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def retry(operation, attempts=3, wait_time=5):
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < attempts - 1:
                time.sleep(wait_time)
            else:
                raise

def insert_data_in_neo4j(df, driver):
    with driver.session() as session:
        # Create credit card nodes
        credit_card_nodes = df[['cc_num', 'first', 'last', 'gender', 'street', 'city', 'state', 'zip', 'lat', 'long', 'city_pop', 'job', 'dob']].drop_duplicates()
        credit_card_query = """
        UNWIND $rows as row
        MERGE (c:CreditCard {cc_num: row.cc_num})
        SET c.first = row.first, c.last = row.last, c.gender = row.gender, c.street = row.street,
            c.city = row.city, c.state = row.state, c.zip = row.zip, c.lat = row.lat,
            c.long = row.long, c.city_pop = row.city_pop, c.job = row.job, c.dob = row.dob
        """
        retry(lambda: session.run(credit_card_query, {'rows': credit_card_nodes.to_dict('records')}))
        print('Inserted credit card nodes')
        
        # Create merchant nodes in batches
        merchant_nodes = df[['merchant', 'merch_lat', 'merch_long']].drop_duplicates()
        merchant_query = """
        UNWIND $rows as row
        MERGE (m:Merchant {name: row.merchant})
        SET m.merch_lat = row.merch_lat, m.merch_long = row.merch_long
        """
        retry(lambda: session.run(merchant_query, {'rows': merchant_nodes.to_dict('records')}))
        print('Inserted merchant nodes')
        
        # Create transaction edges in batches
        transaction_edges = df[['cc_num', 'merchant', 'trans_date_trans_time', 'amt', 'category', 'trans_num', 'is_fraud']]
        transaction_query = """
        UNWIND $rows as row
        MATCH (c:CreditCard {cc_num: row.cc_num}), (m:Merchant {name: row.merchant})
        CREATE (c)-[:TRANSACTION {trans_date_trans_time: row.trans_date_trans_time, amt: row.amt, category: row.category, trans_num: row.trans_num, is_fraud: row.is_fraud}]->(m)
        """
        for batch_data in batch(transaction_edges.to_dict('records'), 50000): 
            retry(lambda: session.run(transaction_query, {'rows': batch_data}))
            print(f"{len(batch_data)} out of {df.shape[0]} edges included")

uri = 'neo4j://localhost:7687'
driver = create_driver(uri, 'neo4j', 'password')

insert_data_in_neo4j(df, driver)

print("Nodes and edges inserted successfully")

driver.close()
