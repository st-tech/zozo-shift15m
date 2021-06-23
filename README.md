# SHIFT28M

We provide the [Datasheet for SHIFT28M](./DATASHEET.md).
This datasheet is based on the [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) template.

## Benchmarks

As templates for numerical experiments on the SHIFT28M dataset, we have published [experimental results for each task with several models](./benchmarks).

## Original Dataset Structure

The original dataset is maintained in jsonl format, and a row consists of the following:

```
{
  "user":{"user_id":"xxxx"},
  "like_num":"xx",
  "set_id":"xxx",
  "items":[
    {"price":"xxxx","item_id":"xxxxxx","category_id1":"xx","category_id2":"xxxxx"},
    ...
  ],
  "publish_date":"yyyy-mm-dd"
}
```



## Contributing
To learn more about making a contribution to SHIFT28M, please see the following materials:
- [Developers Guide](./DEVELOPMENT.md)
- [Task Proposal Guide](./TASK_PROPOSAL.md)
- [Benchmark Proposal Guide](./BENCHMARK.md)

## Citation

```bibtex
@software{https://github.com/st-tech/zr-shift28m,
  url = {https://github.com/st-tech/zr-shift28m},
  author = {Masanari Kimura, Yuki Saito, Kazuya Morishita},
  title = {SHIFT28M: Multiobjective Large-Scale Dataset with Distributional Shifts},
  year = {2021},
}
```

## References
- Gebru, Timnit, et al. "Datasheets for datasets." arXiv preprint arXiv:1803.09010 (2018).
