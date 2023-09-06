from unified_ar import Data, MyTask
from intervaltree.intervaltree import IntervalTree


class MLStrategy(MyTask):

    def train(self):
        pass

    def test(self):
        pass

    def justifySet(self, acts, Train, add_other_action=False):
        inp = [Train]
        out = []
        if (acts[0] != 0):
            acts.insert(0, 0)

        act_map = {a: i for i, a in enumerate(acts)}
        for dtype in inp:
            ndtype = Data(dtype.name)
            ndtype.s_events = dtype.s_events
            ndtype.a_events = dtype.a_events.copy()
            ndtype.a_events.Activity = dtype.a_events.Activity.apply(lambda x: act_map[x] if x in act_map else (len(acts) if add_other_action else 0))

            out.append(ndtype)
            ndtype.act_map = act_map
            ndtype.acts = acts
        return out[0]  # Train
