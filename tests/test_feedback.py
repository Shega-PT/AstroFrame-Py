"""Testes do banco de aprendizagem (recompensa/punição por estrelas)."""

from __future__ import annotations

import pytest

from astroframe.ai.feedback import (
    _PARAM_BOUNDS,
    FeedbackDB,
    apply_learned,
    detection_id,
    nudge_params,
    origin_for,
    profile_for,
    record_run,
)
from astroframe.ai.score import score_from_stars
from astroframe.config import AstroFrameConfig
from astroframe.core.stabilizer import DiskDetection


@pytest.fixture
def db(tmp_path):
    return FeedbackDB(tmp_path / "fb.db")


def _rating(stars: float, metrics: dict | None = None) -> object:
    rating = score_from_stars(stars)
    rating.metrics.update(metrics or {})
    return rating


def test_db_cria_schema_e_ficheiro_restrito(db):
    assert db.path.exists()
    assert db.count() == 0
    assert (db.path.stat().st_mode & 0o777) == 0o600


def test_db_add_run_append_only(db):
    cfg = AstroFrameConfig()
    rating = _rating(4.0, {"background": 1.0, "limb": 1.0, "noise": 0.8, "contrast": 1.0, "reflections": 1.0})
    first = record_run(db, "image", "prof", cfg, origin_for(cfg), rating, source="a.jpg")
    second = record_run(db, "video", "prof", cfg, origin_for(cfg), rating, source="b.mp4")
    assert first.id == 1
    assert second.id == 2
    assert db.count() == 2


def test_db_history_somente_do_perfil_e_ordenada_desc(db):
    cfg = AstroFrameConfig()
    rating = _rating(3.0)
    record_run(db, "image", "alpha", cfg, {}, rating)
    record_run(db, "image", "beta", cfg, {}, rating)
    record_run(db, "image", "alpha", cfg, {}, rating)
    history = db.history("alpha")
    assert [r.id for r in history] == [3, 1]
    assert db.history("beta")[0].id == 2
    assert db.history("gamma") == []


def test_db_round_trip_preserva_campos(db):
    cfg = AstroFrameConfig()
    cfg.denoise.h = 7.5
    rating = _rating(4.0, {"noise": 0.6})
    run = record_run(db, "video", "prof", cfg, origin_for(cfg), rating, stars_user=2.0, source="v.mp4")
    cached = db.history("prof", limit=1)[0]
    assert cached.id == run.id
    assert cached.stars_calc == 4.0
    assert cached.stars_user == 2.0
    assert cached.kind == "video"
    assert cached.source == "v.mp4"
    assert cached.params_used["denoise"]["h"] == 7.5
    assert cached.metrics["noise"] == 0.6
    assert cached.rationale
    assert isinstance(cached.nudge, dict)


def test_nudge_punicao_corrige_direcoes(db):
    cfg = AstroFrameConfig()
    rating = _rating(1.0, {"limb": 0.1, "noise": 0.2, "contrast": 0.3, "background": 0.4, "reflections": 0.5})
    deltas, rationale = nudge_params(rating, cfg)
    assert deltas["denoise.h"] > 0
    assert deltas["unsharp.amount"] > 0
    assert deltas["clahe.clip_limit"] > 0
    assert deltas["polish.feather"] > 0
    assert deltas["polish.corona_scale"] < 0
    assert "punição" in rationale


def test_nudge_recompensa_refina_direcoes(db):
    cfg = AstroFrameConfig()
    rating = _rating(
        4.5, {"limb": 0.95, "noise": 0.9, "background": 1.0, "contrast": 1.0, "reflections": 1.0}
    )
    deltas, rationale = nudge_params(rating, cfg)
    assert deltas["denoise.h"] < 0
    assert deltas["clahe.clip_limit"] < 0
    assert deltas["polish.feather"] < 0
    assert deltas["polish.corona_scale"] > 0
    assert "recompensa" in rationale


def test_nudge_misto_ignora_metricas_boas(db):
    cfg = AstroFrameConfig()
    rating = _rating(3.2, {"limb": 0.4, "noise": 0.9, "background": 1.0, "contrast": 1.0, "reflections": 1.0})
    deltas, _ = nudge_params(rating, cfg)
    assert deltas.get("denoise.h", 0) >= 0  # punição do limb, sem recompensa do noise
    assert "clahe.clip_limit" not in deltas


def test_nudge_sem_metricas_nao_altera(db):
    cfg = AstroFrameConfig()
    deltas, rationale = nudge_params(_rating(2.0), cfg)
    assert deltas == {}
    assert "sem alterações" in rationale


def test_nudge_respeita_limites(db):
    cfg = AstroFrameConfig()
    cfg.denoise.h = _PARAM_BOUNDS["denoise.h"][1] - 0.05
    rating = _rating(1.0, {"noise": 0.1})
    record_run(db, "image", "prof", cfg, {}, rating)
    adjusted = apply_learned(cfg, "prof", db=db)
    assert adjusted.denoise.h <= _PARAM_BOUNDS["denoise.h"][1]


def test_apply_learned_sem_nudges_devolve_mesmo_valores(db):
    cfg = AstroFrameConfig()
    adjusted = apply_learned(cfg, "desconhecido", db=db)
    assert adjusted.clahe.clip_limit == cfg.clahe.clip_limit


def test_apply_learned_aplica_delta_sem_mutar_original(db):
    cfg = AstroFrameConfig()
    cfg.denoise.h = 4.0
    rating = _rating(1.0, {"noise": 0.1, "limb": 0.5})
    record_run(db, "image", "prof", cfg, {}, rating)
    adjusted = apply_learned(cfg, "prof", db=db)
    assert adjusted.denoise.h != cfg.denoise.h
    assert cfg.denoise.h == 4.0


def test_apply_learned_persiste_entre_execucoes(db):
    cfg = AstroFrameConfig()
    rating = _rating(1.0, {"noise": 0.1, "limb": 0.5, "background": 0.5, "contrast": 0.5, "reflections": 0.5})
    record_run(db, "image", "prof", cfg, {}, rating)
    first = apply_learned(cfg, "prof", db=db)
    second = apply_learned(first, "prof", db=db)
    assert first.denoise.h >= cfg.denoise.h
    assert abs(second.denoise.h - first.denoise.h) > 0


def test_estrelas_manuais_tem_peso_duplo(db):
    cfg = AstroFrameConfig()
    base = record_run(db, "image", "prof", cfg, {}, _rating(1.0, {"noise": 0.1}))
    manual = record_run(db, "image", "prof2", cfg, {}, _rating(1.0, {"noise": 0.1}), stars_user=1.0)
    auto_delta = base.nudge["denoise.h"]
    manual_delta = manual.nudge["denoise.h"]
    assert manual_delta == pytest.approx(auto_delta * 2.0)


def test_profile_for_grupos_estaveis():
    assert profile_for("image", 480, 360) == profile_for("image", 480, 360)
    assert profile_for("image", 480, 360) != profile_for("video", 480, 360)
    assert profile_for("image", 1920, 1080) != profile_for("image", 480, 360)
    assert profile_for("image", 480, 360, camera="A") != profile_for("image", 480, 360, camera="B")
    assert profile_for("image", 480, 360, iso=3200) != profile_for("image", 480, 360, iso=100)


def test_origin_for_apenas_campos_relevantes():
    cfg = AstroFrameConfig()
    origin = origin_for(cfg)
    assert origin["denoise.h"] == 5.0
    assert origin["polish.corona_scale"] == 1.6
    assert len(origin) == 4


def test_detection_id_estavel():
    assert detection_id(DiskDetection(10, 20, 30)) == "10x20r30"
    assert detection_id(DiskDetection(10, 20, 30)) == detection_id(DiskDetection(10, 20, 30))
    assert detection_id(None) == "none"


def test_db_path_expande_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    db = FeedbackDB("~/sub/feedback.db")
    assert db.path == tmp_path / "sub" / "feedback.db"
    assert db.path.exists()


def test_record_run_rating_manual_sem_metricas(db):
    cfg = AstroFrameConfig()
    run = record_run(db, "image", "prof", cfg, {}, score_from_stars(3.0))
    assert run.stars_calc == 3.0
    assert run.nudge == {}


def test_db_chmod_falha_nao_impede_uso(tmp_path, monkeypatch):
    import pathlib

    def raise_oserror(self, *args, **kwargs):
        raise OSError("sem permissões")

    monkeypatch.setattr(pathlib.Path, "chmod", raise_oserror)
    db = FeedbackDB(tmp_path / "fb.db")
    assert db.count() == 0


def test_apply_learned_respeita_limite_inferior(db):
    cfg = AstroFrameConfig()
    cfg.denoise.h = _PARAM_BOUNDS["denoise.h"][0] + 0.05
    rating = _rating(
        4.5, {"noise": 0.95, "limb": 0.9, "background": 1.0, "contrast": 1.0, "reflections": 1.0}
    )
    record_run(db, "image", "prof", cfg, {}, rating)
    adjusted = apply_learned(cfg, "prof", db=db)
    assert adjusted.denoise.h >= _PARAM_BOUNDS["denoise.h"][0]


def test_default_db_path_relativo_resolve_contra_data_root(monkeypatch, tmp_path):
    from astroframe import paths

    monkeypatch.delenv("ASTROFRAME_FEEDBACK_DB", raising=False)
    monkeypatch.setenv("ASTROFRAME_DATA_DIR", str(tmp_path))
    db = FeedbackDB()
    assert db.path == paths.data_root() / "logs/system/feedback.db"
    assert db.path.exists()


def test_apply_learned_usa_banco_por_omissao(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTROFRAME_FEEDBACK_DB", str(tmp_path / "default.db"))
    db = FeedbackDB()
    cfg = AstroFrameConfig()
    cfg.denoise.h = 4.0
    record_run(db, "image", "prof", cfg, {}, _rating(1.0, {"noise": 0.1}))
    adjusted = apply_learned(cfg, "prof")
    assert adjusted.denoise.h > 4.0


def test_nudge_metricas_desconhecidas_ignoradas(db):
    cfg = AstroFrameConfig()
    deltas, rationale = nudge_params(_rating(1.0, {"weird": 0.1}), cfg)
    assert deltas == {}
    assert "sem alterações" in rationale


def test_record_run_fallback_sem_historico():
    class StubDB:
        def add_run(self, **kwargs):
            return 7

        def history(self, profile, limit=1):
            return []

    cfg = AstroFrameConfig()
    run = record_run(StubDB(), "image", "perfil", cfg, {}, _rating(3.0, {}))
    assert run.id == 7
    assert run.profile == "perfil"
    assert run.stars_calc == 3.0


# ----------------------------------------------------------------- logs ----


def test_db_log_grava_e_lista(db):
    db.log("info", "validator", "série 1 concluída", {"round": 1})
    db.log("warn", "validator", "sem patches")
    db.log("info", "enhancer", "campeão promovido")
    rows = db.logs()
    assert [r["message"] for r in rows] == ["campeão promovido", "sem patches", "série 1 concluída"]
    filtered = db.logs(component="validator")
    assert len(filtered) == 2
    assert filtered[0]["level"] == "warn"
    assert filtered[0]["details"] is None
    assert filtered[1]["details"] == {"round": 1}
    limited = db.logs(limit=1)
    assert len(limited) == 1


def test_db_log_aceita_qualquer_nivel(db):
    db.log("debug", "componente", "mensagem mínima")
    assert len(db.logs()) == 1


# ------------------------------------------------------------ modelos/campeão --


def test_db_add_model_primeiro_e_campeao(db):
    result = db.add_model("disk_filter", "/tmp/m1.npz", {"score": 80.0, "accuracy": 0.9}, dataset_size=10)
    assert result["promoted"] is True
    assert result["previous"] is None
    assert result["champion"]["path"] == "/tmp/m1.npz"
    champion = db.champion("disk_filter")
    assert champion["metrics"]["score"] == 80.0
    assert champion["dataset"] == 10
    assert champion["promoted"] is True


def test_db_add_model_so_promove_estritamente_melhor(db):
    db.add_model("disk_filter", "/tmp/m1.npz", {"score": 80.0})
    igual = db.add_model("disk_filter", "/tmp/m2.npz", {"score": 80.0})
    assert igual["promoted"] is False
    pior = db.add_model("disk_filter", "/tmp/m3.npz", {"score": 70.0})
    assert pior["promoted"] is False
    melhor = db.add_model("disk_filter", "/tmp/m4.npz", {"score": 90.0})
    assert melhor["promoted"] is True
    assert db.champion("disk_filter")["path"] == "/tmp/m4.npz"


def test_db_add_model_metricas_por_tipo_e_campeoes_independentes(db):
    db.add_model("disk_filter", "/tmp/d1.npz", {"score": 50.0})
    db.add_model("enhancer", "/tmp/e1.npz", {"mean_delta": 0.5})
    db.add_model("enhancer", "/tmp/e2.npz", {"mean_delta": 0.9})
    assert db.champion("disk_filter")["path"] == "/tmp/d1.npz"
    assert db.champion("enhancer")["path"] == "/tmp/e2.npz"


def test_db_add_model_metric_alternativa(db):
    result = db.add_model("disk_filter", "/tmp/m1.npz", {"accuracy": 0.95}, metric_name="accuracy")
    assert result["promoted"] is True


def test_db_add_model_metrica_em_falta_levanta(db):
    with pytest.raises(ValueError, match="Métrica 'score' ausente"):
        db.add_model("disk_filter", "/tmp/m1.npz", {"accuracy": 0.5})
    with pytest.raises(ValueError, match="Métrica 'mean_delta' ausente"):
        db.add_model("enhancer", "/tmp/e1.npz", {"mse": 0.1})


def test_db_champion_vazio_e_por_tipo(db):
    assert db.champion("disk_filter") is None
    assert db.champion("desconhecido") is None


def test_db_model_history_ordenada_desc_com_limite(db):
    for i in range(5):
        db.add_model("disk_filter", f"/tmp/m{i}.npz", {"score": float(i)})
    history = db.model_history("disk_filter", limit=3)
    assert [h["id"] for h in history] == [5, 4, 3]
    assert history[0]["promoted"] is True
    assert db.model_history("enhancer") == []
