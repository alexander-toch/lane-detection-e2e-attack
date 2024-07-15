import sqlite3

class ExperimentDatabase:
    def __init__(self, db):
        self.conn = sqlite3.connect(db)
        self.cur = self.conn.cursor()
        self.cur.execute("CREATE TABLE IF NOT EXISTS experiment ("
                        "id INTEGER PRIMARY KEY," 
                        "start_time TIMESTAMP,"
                        "end_time TIMESTAMP,"
                        "attack_type TEXT,"
                        "num_scenarios INTEGER)"
        )
        self.cur.execute("CREATE TABLE IF NOT EXISTS simulation ("
                        "id INTEGER PRIMARY KEY,"
                        "experiment_id INTEGER,"
                        "seed INTEGER,"
                        "map_config TEXT,"
                        "max_steps INTEGER,"
                        "start_time TIMESTAMP,"
                        "end_time TIMESTAMP,"
                        "attack_active BOOLEAN,"
                        "patch_geneneration_iterations INTEGER,"
                        "attack_at_meter REAL,"
                        "simulator_width INTEGER,"
                        "simulator_height INTEGER,"
                        "lane_detection_model TEXT,"
                        "end_reason TEXT,"
                        "end_step INTEGER,"
                        "end_meter REAL,"
                        "e2e_ld REAL,"
                        "FOREIGN KEY (experiment_id) REFERENCES experiment (id))"
        )
        self.cur.execute("CREATE TABLE IF NOT EXISTS simulation_step ("
                        "id INTEGER PRIMARY KEY,"
                        "simulation_id INTEGER,"
                        "step INTEGER,"
                        "time TIMESTAMP,"
                        "offset_center REAL,"
                        "offset_center_simulator REAL,"
                        "car_position_x_meter REAL,"
                        "steering REAL,"
                        "speed REAL,"
                        "throttle_brake REAL,"
                        "FOREIGN KEY (simulation_id) REFERENCES simulation (id))"
        )
        self.conn.commit()

    def add_experiment(self, start_time, end_time, attack_type, num_scenarios):
        self.cur.execute("INSERT INTO experiment VALUES (NULL, ?, ?, ?, ?)",
                            (start_time, end_time, attack_type, num_scenarios))
        self.conn.commit()
        return self.cur.lastrowid
    
    def finish_experiment(self, experiment_id, end_time):
        self.cur.execute("UPDATE experiment SET end_time = ? WHERE id = ?", (end_time, experiment_id))
        self.conn.commit()
    
    def add_simulation(self, experiment_id, seed, map_config, max_steps, start_time, end_time, attack_active, 
                       patch_geneneration_iterations, attack_at_meter, simulator_width, simulator_height, 
                       lane_detection_model, end_reason, end_step, end_meter, e2e_ld):
        self.cur.execute("INSERT INTO simulation VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (experiment_id, seed, map_config, max_steps, start_time, end_time, attack_active, 
                             patch_geneneration_iterations, attack_at_meter, simulator_width, simulator_height, 
                             lane_detection_model, end_reason, end_step, end_meter, e2e_ld))
        self.conn.commit()
        return self.cur.lastrowid
    
    def add_simulation_steps(self, simulation_id, steps):
        insert_tuples = [(simulation_id, step['step'], step['time'], step['offset_center'], step['offset_center_simulator'], 
                          step['car_position_x_meter'], step['steering'], step['speed'], step['throttle_brake']) for step in steps]
        self.cur.executemany('INSERT INTO simulation_step VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?)', insert_tuples)
        self.conn.commit()



    def __del__(self):
        self.conn.close()