import pandas as pd
import gokart
import luigi


class MakePairedData(gokart.TaskOnKart):
    task_namespace = 'novelty_enhanced_bpr'

    click_task = gokart.TaskInstanceParameter()
    positive_sample_weight: int = luigi.IntParameter()
    distance_threshold: float = luigi.FloatParameter()

    def requires(self):
        return self.click_task

    def run(self):
        clicks = self.load()['clicks_train']
        item_distance = self.load()['item_distance']
        paired_data = self._run(clicks, item_distance, self.positive_sample_weight, self.distance_threshold)
        self.dump(paired_data)

    @staticmethod
    def _run(clicks: pd.DataFrame, item_distance: pd.DataFrame, positive_sample_weight: int, distance_threshold: float) -> pd.DataFrame:
        clicked_data = clicks[clicks['click'].astype(bool)].rename(columns={'item_id': 'positive_item_id'})
        not_clicked_data = clicks[~clicks['click'].astype(bool)].rename(columns={'item_id': 'negative_item_id'})

        not_clicked_data = not_clicked_data.groupby('user_id').apply(
            lambda x: x.sample(positive_sample_weight)).reset_index(drop=True)

        paired_data = pd.merge(clicked_data[['user_id', 'positive_item_id']],
                               not_clicked_data[['user_id', 'negative_item_id']],
                               on='user_id', how='inner')

        paired_data = pd.merge(paired_data, item_distance, left_on=['positive_item_id', 'negative_item_id'],
                               right_on=['item_id_x', 'item_id_y'], how='inner')
        if distance_threshold:
            paired_data = paired_data[paired_data['distance'] < distance_threshold]

        return paired_data[['user_id', 'positive_item_id', 'negative_item_id']]
