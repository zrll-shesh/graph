
CREATE CONSTRAINT dosen_name IF NOT EXISTS FOR (d:Dosen) REQUIRE d.nama_dosen IS UNIQUE;
CREATE CONSTRAINT topic_id IF NOT EXISTS FOR (t:Topic) REQUIRE t.topic_id IS UNIQUE;

LOAD CSV WITH HEADERS FROM 'file:///nodes_dosen.csv' AS row
MERGE (d:Dosen {nama_dosen: row.nama_dosen})
SET d.dosen_id = row.dosen_id,
    d.prodi = row.prodi,
    d.n_papers_total = toInteger(row.n_papers_total),
    d.thematic_cluster = toInteger(row.thematic_cluster),
    d.thematic_cluster_name = row.thematic_cluster_name,
    d.structural_community = toInteger(row.structural_community),
    d.n_community_memberships = toInteger(row.n_community_memberships),
    d.constraint = toFloat(row.constraint),
    d.betweenness = toFloat(row.betweenness),
    d.broker_typology = row.broker_typology,
    d.composite_rank_score = toFloat(row.composite_rank_score);

LOAD CSV WITH HEADERS FROM 'file:///nodes_topic.csv' AS row
MERGE (t:Topic {topic_id: toInteger(row.topic)})
SET t.top_words = row.top_words,
    t.topic_name = row.topic_name,
    t.n_dosen_dominant = toInteger(row.n_dosen_dominant);

LOAD CSV WITH HEADERS FROM 'file:///edges_collaboration.csv' AS row
MATCH (a:Dosen {nama_dosen: row.source})
MATCH (b:Dosen {nama_dosen: row.target})
MERGE (a)-[r:COLLABORATES_WITH]-(b)
SET r.weight = toInteger(row.weight);

LOAD CSV WITH HEADERS FROM 'file:///edges_dosen_topic.csv' AS row
MATCH (d:Dosen {nama_dosen: row.dosen})
MATCH (t:Topic {topic_id: toInteger(row.topic)})
MERGE (d)-[r:RESEARCHES]->(t)
SET r.weight = toFloat(row.weight);
