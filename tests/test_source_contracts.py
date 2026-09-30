from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def read(relative):
    return (ROOT / relative).read_text(encoding="utf-8")


def test_dataset_generation_contract():
    registry = yaml.safe_load(read("configs/datasets.yaml"))
    assert registry["defaults"]["num_views"] == 120
    assert registry["defaults"]["n_conformers"] == 5
    source = read("src/cgim/data/generator.py")
    assert "self._get_masked_indices(aug_mol, 0.0, 0.15)" in source
    assert "random.uniform(-15, 15)" in source


def test_pretrain_parameter_contract():
    config = yaml.safe_load(read("configs/pretrain.yaml"))
    assert config["sample_size"] == 50000
    assert config["sample_seed"] == 42
    assert config["pseudo_labels"]["clusters"] == [100, 1000]
    assert config["regression_target"]["components"] == 64
    assert config["loader"]["batch_size"] == 64
    assert config["training"]["learning_rate"] == 0.001
    assert config["training"]["temperature"] == 0.1
    assert config["training"]["early_stopping_patience"] == 10
    assert config["training"]["gradient_clip"] == 0.1


def test_finetune_parameter_contract():
    for name in ("bbbp", "lightbbb", "b3db"):
        config = yaml.safe_load(read(f"configs/finetune_{name}.yaml"))
        assert config["seed"] == 43
        assert config["epochs"] == 500
        assert config["learning_rate"] == 0.001
        assert config["scheduler_factor"] == 0.5
        assert config["scheduler_patience"] == 3
        assert config["early_stopping_patience"] == 10
        assert config["gradient_clip"] == 0.1


def test_loss_equation_contracts_are_present():
    source = read("src/cgim/losses.py")
    assert "weight = 2 - batch_size * nnf.softmax(similarities / sigma, dim=0)" in source
    assert "loss = (loss1 + loss2) / 2" in source
    assert "sim_matrix = sim_matrix.masked_fill(mask, -1e9)" in source
    assert "loss = -torch.log(pos_sum / (total_sum + 1e-9))" in source


def test_split_and_metric_contracts_are_present():
    splitters = read("src/cgim/data/splitters.py")
    assert "key=lambda x: (len(x[1]), x[1][0])" in splitters
    assert "if len(train_inds) + len(scaffold_set) > train_cutoff" in splitters
    metrics = read("src/cgim/metrics.py")
    assert "probs = torch.sigmoid(logits)" in metrics
    assert "preds = (probs >= 0.5).long()" in metrics
    assert "AveragePrecision(task='binary')" in metrics
    assert "CohenKappa(task='binary', weights='quadratic')" in metrics


def test_active_objectives_are_unchanged():
    finetune = read("src/cgim/training/finetune.py")
    assert "loss_cls = 0.5 * (criterion_cls(logit1, labels)" in finetune
    assert "+ criterion_cls(logit2, labels))" in finetune
    pretrain = read("src/cgim/training/pretrain.py")
    assert "reconstruction_loss = 0.5 * (" in pretrain
    assert "criterion_reg(rec1, scaffold_img) + criterion_reg(rec2, scaffold_img)" in pretrain
    assert "regression_loss = 0.5 * (" in pretrain
    assert "criterion_reg(reg1, regression_vec) + criterion_reg(reg2, regression_vec)" in pretrain
    assert "loss = contrastive_loss + cls_loss + reconstruction_loss + regression_loss" in pretrain


def test_pretraining_validation_is_loss_only():
    source = read("src/cgim/training/pretrain.py")
    assert 'mode="min"' in source
    assert 'best_valid_loss = float("inf")' in source
    assert "valid_loss < best_valid_loss" in source
    assert "early_stopping_patience=10" in source
    assert "AUROC" not in source
    assert "test_loader" not in source
    assert "warmup" not in source


def test_downstream_first_view_behavior_is_explicit():
    source = read("src/cgim/data/loaders.py")
    assert "is_validation=True" in source


def test_direct_pipeline_entry_files_are_present():
    pipelines = {
        "src/generate_dataset.py": "generate_dataset(",
        "src/pretrain.py": "train_model(",
        "src/finetune.py": "finetune_model_bce(",
        "src/evaluate.py": "evaluate_repeated(",
    }
    for relative, direct_call in pipelines.items():
        source = read(relative)
        assert 'if __name__ == "__main__"' in source
        assert direct_call in source


def test_no_console_command_layer():
    assert not (ROOT / "src/cgim/cli").exists()
    assert "[project.scripts]" not in read("pyproject.toml")


def test_pipeline_arguments_and_gradcam_output_are_available():
    for relative in (
        "src/generate_dataset.py",
        "src/pretrain.py",
        "src/finetune.py",
        "src/evaluate.py",
        "src/gradcam.py",
    ):
        assert "argparse.ArgumentParser" in read(relative)
    plotting = read("src/cgim/plotting.py")
    assert '("svg", "png")' in plotting
    assert "figure.savefig(" in plotting
    assert "def save_gradcam(" in plotting
    assert "T" + "SNE" not in plotting
    assert "save_" + "embedding_plots" not in plotting


def test_training_arguments_use_checkpoint_vocabulary():
    rejected_aliases = ("res" + "ume", "workers", "batch", "lr")
    for relative in ("src/pretrain.py", "src/finetune.py", "src/evaluate.py"):
        source = read(relative)
        assert '"--checkpoint"' in source
        for name in rejected_aliases:
            assert f'"--{name}"' not in source


def test_generated_image_paths_are_portable():
    generator = read("src/cgim/data/generator.py")
    preparation = read("src/cgim/data/preparation.py")
    assert "path.relative_to(self.path_root).as_posix()" in generator
    assert 'df["image_path"] = df["image_path"].map(resolve)' in preparation
