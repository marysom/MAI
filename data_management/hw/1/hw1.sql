CREATE TABLE films (
    title TEXT
  , id BIGINT NOT NULL
  , country TEXT
  , box_office DECIMAL(25,0)
  , releas_date DATE
  , PRIMARY KEY (id) 
) 
;

CREATE TABLE person (
    id BIGINT NOT NULL
  , fio TEXT
  , PRIMARY KEY (id)
)
;

CREATE TABLE person2content (
    person_id BIGINT
  , film_id BIGINT
  , person_type TEXT
)
;

INSERT INTO films VALUES('Harry Potter and the Goblet', 20001, 'UK', '896911785', '2005-11-06');
INSERT INTO films VALUES('Fight club', 20002, 'USA', '100853753', '1999-09-10');
INSERT INTO films VALUES('Once Upon a Time ... in Hollywood', 20003, 'USA', '58798986', '2019-05-21');
INSERT INTO films VALUES('The Great Gatsby', 20004, 'Australia', '351040415', '2013-05-01');
INSERT INTO films VALUES('The Curious Case of Benjamin Button', 20005, 'USA', '333932083', '2008-12-10');

INSERT INTO person VALUES(400001, 'Somova Maria Sergeevna');
INSERT INTO person VALUES(400002, 'Silyakova Alena Sergeevna');
INSERT INTO person VALUES(400003, 'Koltakov Ivan Pavlovich');

INSERT INTO person2content VALUES(400001, 20004, 'actor');
INSERT INTO person2content VALUES(400003, 20002, 'producer');
INSERT INTO person2content VALUES(400002, 20005, 'actor');
