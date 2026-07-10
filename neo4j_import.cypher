
// Skrip import Neo4j (jalankan via neo4j-admin database import, atau LOAD CSV)
// 1) Via neo4j-admin (paling cepat utk data besar):
//    neo4j-admin database import full --nodes=Dosen=neo4j_nodes.csv --relationships=COLLABORATES_WITH=neo4j_edges.csv neo4j
//
// 2) Via Cypher LOAD CSV (utk update incremental pd DB yg sudah jalan):
LOAD CSV WITH HEADERS FROM 'file:///neo4j_nodes.csv' AS row
MERGE (d:Dosen {nama_norm: row.`nama_norm:ID`})
SET d.prodi = row.prodi,
    d.total_paper = toInteger(row.`total_paper:int`),
    d.composite_score = toFloat(row.`composite_score:float`),
    d.cluster_consensus = toInteger(row.`cluster_consensus:int`),
    d.broker_score = toFloat(row.`broker_score:float`);

LOAD CSV WITH HEADERS FROM 'file:///neo4j_edges.csv' AS row
MATCH (a:Dosen {nama_norm: row.`:START_ID`}), (b:Dosen {nama_norm: row.`:END_ID`})
MERGE (a)-[r:COLLABORATES_WITH]-(b)
SET r.weight = toInteger(row.`weight:int`);

// Contoh query eksplorasi setelah import:
// Cari 10 dosen dgn composite_score tertinggi:
// MATCH (d:Dosen) RETURN d.nama_norm, d.composite_score ORDER BY d.composite_score DESC LIMIT 10;
// Cari broker antar cluster:
// MATCH (d:Dosen) WHERE d.broker_score > 0.85 RETURN d.nama_norm, d.cluster_consensus, d.broker_score ORDER BY d.broker_score DESC;
