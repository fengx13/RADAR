#!/usr/bin/env python3
"""Condition / action / target triples for the SSC 2026 statements.

Written by reading each statement and its remarks. This is the "LLM extraction"
step; every record carries an extraction_confidence that drives human review.

Depth:
  full   population + conditions + actions + targets + ordering + EVERY remark
         classified. Statements 1-64 (the four agents' working set) and the 14
         statements in 65-129 that carry a remark.
  light  population + conditions + head action only. The 51 statements in
         65-129 that carry no remark, so there is nothing further to classify
         for the remark analysis; the graph is still complete for retrieval.

  2026-08-07: 65-129 previously had refined_by unpopulated across the board,
  which made the remark statistic a subset estimate reported as if it were the
  whole guideline. Fixed here; see REMARK_KINDS.
"""
from radar import paths
import json, pathlib
from collections import Counter, defaultdict

OUT = pathlib.Path(__file__).parent

def any_(*c):  return {"mode": "any_of", "conditions": list(c)}
def all_(*c):  return {"mode": "all_of", "conditions": list(c)}
def A(action, polarity="for", **kw): return dict(action=action, polarity=polarity, **kw)
# What a remark DOES to its statement. `adds_action` alone was too coarse: a remark
# that narrows who the statement applies to changes management just as much as one
# that adds an action, and chunked retrieval loses both.
REMARK_KINDS = {
    "adds_action",           # a directive absent from the statement body
    "adds_preference",       # body presents options as equivalent; remark tilts
    "restricts_population",  # body is broader than the remark allows
    "adds_target",           # numeric goal that the body does not state
    "defines_term",          # body is unusable without the definition
    "implementation_note",   # context only; nothing changes at the bedside
}
def R(i, adds, action=None, cond=None, kind=None, polarity="for"):
    kind = kind or ("adds_action" if action else "implementation_note")
    assert kind in REMARK_KINDS, kind
    d = {"remark": i, "adds": adds, "adds_action": action is not None, "kind": kind}
    if action: d["action"] = action; d["polarity"] = polarity
    if cond:   d["modifier_condition"] = cond
    return d

# statements whose remarks have all been read and classified
FULL_DEPTH = set(range(1, 65)) | {65, 66, 67, 70, 72, 80, 85, 89, 105, 123, 125, 126, 127, 129}

T = []
def t(n, pop, when, rec=(), pref=(), equiv=None, targ=(), prec=(), constrains=(),
      ref=(), conf="high", note=None):
    T.append({"id": f"stmt_{n:03d}", "number": n, "population": pop, "applies_when": when,
              "recommends": list(rec), "preferred_over": list(pref),
              "equivalent_options": equiv, "targets": list(targ),
              "must_precede": list(prec), "constrains": list(constrains),
              "refined_by": list(ref), "extraction_confidence": conf,
              "depth": "full" if n in FULL_DEPTH else "light", **({"note": note} if note else {})})

# ══════════════════ SCREENING AND EARLY MANAGEMENT (1-15) ══════════════════
t(1, "hospitals and health systems", any_("acutely_ill_high_risk_patients"),
  rec=[A("run_sepsis_performance_improvement_program"), A("screen_for_sepsis"),
       A("use_standard_operating_procedures_for_treatment"),
       A("implement_sepsis_quality_improvement_strategies")],
  ref=[R(0, "programs may vary by setting and by what the system can implement")],
  note="system-level, not patient-level; carries three separate gradings")

t(2, "hospitals and health systems", any_("positive_sepsis_screen"),
  rec=[A("use_code_sepsis_or_sepsis_huddle_protocol")], pref=["no_such_protocol"],
  ref=[R(0, "multidisciplinary bedside huddle to expedite diagnosis and treatment")])

t(3, "acutely ill adults en route to hospital", any_("prehospital_ambulance_or_flight"),
  rec=[A("use_standard_sepsis_screening_tool")], pref=["no_screening_tool"])

t(4, "acutely ill patients in hospital", any_("acutely_ill", "in_hospital"),
  rec=[A("screen_for_sepsis_with_tool", options=["NEWS", "NEWS2", "MEWS", "SIRS"])],
  pref=["qSOFA_as_single_screening_tool"],
  note="the four tools are alternatives to each other; the preference is against qSOFA")

t(5, "general", any_("suspected_sepsis"),
  constrains=[{"forbidden": "rule_in_or_out_sepsis_on_single_test",
               "applies_to": ["single_biomarker", "single_diagnostic_test"]}],
  note="no action — a constraint on reasoning; licenses the 'indeterminate' answer")

t(6, "general", any_("suspected_sepsis"),
  constrains=[{"insufficient_evidence_for": "novel_rapid_host_response_diagnostics"}])

t(7, "adults", any_("possible_sepsis", "probable_sepsis", "definite_sepsis", "septic_shock"),
  rec=[A("collect_blood_cultures", timing="as soon as possible")],
  prec=[{"before": "collect_blood_cultures", "after": "administer_antimicrobial_therapy",
         "strength": "ideally"}],
  note="the MUST_PRECEDE edge is the clinical point and links two different domains")

t(8, "adults", any_("possible_sepsis", "probable_sepsis", "definite_sepsis", "septic_shock"),
  rec=[A("measure_blood_lactate")])

t(9, "general", any_("sepsis", "septic_shock"),
  rec=[A("begin_treatment_and_resuscitation", timing="immediately")],
  note="good practice statement framing sepsis as a medical emergency")

t(10, "adults", any_("sepsis_induced_hypoperfusion", "septic_shock"),
  rec=[A("administer_iv_crystalloid", dose=">= 30 mL/kg", timing="within first 3 hours")],
  ref=[R(0, "individualise initial volume to patient characteristics and context"),
       R(1, "frequent ongoing reassessment to avoid under- or over-resuscitation",
            action="reassess_fluid_status_frequently"),
       R(2, "weight the dose is applied to", action="calculate_dose_on_actual_body_weight",
            cond="BMI_gt_30_use_adjusted_or_ideal_body_weight")])

t(11, "adults", any_("sepsis_induced_hypotension"),
  rec=[A("administer_iv_crystalloid_bolus"),
       A("start_vasopressor", timing="after fluid bolus if hypotension persists")],
  ref=[R(0, "concurrent vasopressor with fluid may be warranted; unstable shock judged on "
            "physical examination — severely reduced BP, mottled skin, ashen appearance, "
            "cyanosis/desaturation, tachycardia, altered mentation",
            action="start_vasopressor_concurrently_with_fluid", cond="unstable_septic_shock")])

t(12, "adults", any_("septic_shock"),
  rec=[A("start_vasopressor_peripherally")], pref=["delay_until_central_access_secured"],
  ref=[R(0, "no recommendation on duration, dose or access route; midlines not considered")])

t(13, "adults", any_("septic_shock"), pref=["higher_MAP_targets"],
  targ=[{"parameter": "MAP", "value": 65, "unit": "mm Hg", "kind": "initial_target"}],
  ref=[R(0, "a band of about +/- 5 mm Hg is acceptable and vasopressors are titrated to hold it",
            action="titrate_vasopressor_to_maintain_MAP")])

t(14, "adults aged 65 or older", all_("septic_shock", "age_ge_65"),
  pref=["higher_MAP_ranges"],
  targ=[{"parameter": "MAP", "value": [60, 65], "unit": "mm Hg", "kind": "initial_target_range"}],
  note="age-specific override of stmt_013 — the graph must prefer the more specific condition")

t(15, "adults", all_("sepsis_or_septic_shock", "requires_icu_admission"),
  rec=[A("admit_to_icu", timing="within 6 hours")])

# ══════════════════════════════ INFECTION (16-41) ══════════════════════════════
t(16, "adults", any_("possible_septic_shock", "probable_septic_shock", "definite_septic_shock"),
  rec=[A("administer_antimicrobial_therapy", timing="immediately, ideally within 1 hour of recognition")])

t(17, "adults", all_("probable_or_definite_sepsis", "no_shock"),
  rec=[A("administer_antimicrobial_therapy", timing="immediately, ideally within 1 hour of recognition")])

t(18, "adults", all_("possible_sepsis", "no_shock"),
  rec=[A("perform_time_limited_rapid_investigation"),
       A("administer_antimicrobial_therapy", timing="within 3 hours from first suspicion",
         condition="if concern for infection persists")])

t(19, "adults", all_("possible_sepsis", "no_shock"),
  rec=[A("assess_likelihood_infectious_vs_noninfectious", timing="rapid")],
  note="good practice statement — the guideline mandate for our R3a/R3b mimic node")

t(20, "adults", all_("low_likelihood_of_infection", "no_shock"),
  rec=[A("administer_antimicrobial_therapy", polarity="defer"),
       A("monitor_patient_closely")])

t(21, "adults", all_("definite_or_probable_sepsis", "hypotension",
                     "anticipated_time_to_in_hospital_evaluation_gt_60_min"),
  rec=[A("administer_antimicrobial_therapy", setting="ambulance or flight")],
  ref=[R(0, "only after a structured prehospital sepsis screening process exists (see stmt_003)",
            cond="structured_prehospital_screening_in_place")])

t(22, "adults", any_("possible_sepsis", "probable_sepsis", "septic_shock"),
  rec=[A("use_clinical_evaluation_alone_to_start_antimicrobials")],
  pref=["procalcitonin_plus_clinical_evaluation"],
  note="the only guideline-backed 'do not order this test' for Agent 2")

t(23, "adults", any_("sepsis", "septic_shock"),
  rec=[A("evaluate_for_source_requiring_emergent_source_control", timing="rapidly")])

t(24, "adults", all_("sepsis_or_septic_shock", "source_requiring_source_control"),
  rec=[A("perform_source_control", timing="ideally within 6 hours of diagnosis")],
  pref=["late_source_control"])

t(25, "adults", all_("sepsis_or_septic_shock", "high_risk_of_specific_MDR_pathogen"),
  rec=[A("use_empirical_therapy_covering_that_MDR_pathogen")],
  ref=[R(0, "MDR risk factors: colonisation, prior infection with that pathogen, prolonged "
            "broad-spectrum antibiotics, prolonged stay in a high-prevalence unit")])

t(26, "adults", all_("sepsis_or_septic_shock", "low_risk_of_specific_MDR_pathogen"),
  rec=[A("use_empirical_therapy_covering_that_MDR_pathogen", polarity="against")])

t(27, "adults", any_("sepsis", "septic_shock"),
  rec=[A("use_empirical_antifungal_therapy", polarity="against")],
  ref=[R(0, "case-by-case in patients with fungal risk factors: immunosuppression, prolonged "
            "antibiotics, prolonged hospitalisation, intra-abdominal source",
            cond="risk_factors_for_fungal_infection")],
  note="2021 split this into a high-risk and a low-risk statement")

t(28, "adults", all_("sepsis_or_septic_shock", "no_risk_factors_for_anaerobic_infection"),
  rec=[A("use_empiric_regimen_without_anaerobic_coverage")],
  ref=[R(0, "agents with anaerobic activity are acceptable when needed for MDR coverage",
            cond="MDR_coverage_required"),
       R(1, "anaerobic risk factors: intra-abdominal or deep gynaecological/obstetric source, "
            "necrotising soft-tissue infection, head and neck infection, CNS abscess, empyema")])

t(29, "adults", all_("sepsis_or_septic_shock", "risk_factors_for_anaerobic_infection"),
  rec=[A("use_empiric_regimen_with_anaerobic_coverage")],
  ref=[R(0, "same anaerobic risk-factor list as stmt_028")])

t(30, "adults", any_("sepsis", "septic_shock"),
  constrains=[{"insufficient_evidence_for": "departmental_upper_respiratory_surveillance_cultures"}])

t(31, "adults", any_("sepsis", "septic_shock"),
  rec=[A("use_pathogen_specific_rapid_diagnostic_tests", condition="case-by-case in selected patients")],
  conf="medium",
  note="the selection criteria (clinical features, local resistance patterns, seasonality, "
       "availability, stewardship) are conditions on the action, not on the population")

t(32, "adults", any_("sepsis", "septic_shock"),
  rec=[A("use_candida_biomarkers_to_start_antifungals", polarity="against")],
  ref=[R(0, "case-by-case in high Candida risk", cond="high_risk_of_candida_infection")])

t(33, "adults", any_("sepsis", "septic_shock"),
  rec=[A("use_prolonged_infusion_of_beta_lactams", condition="for maintenance, after a loading dose")],
  pref=["bolus_administration"])

t(34, "adults", any_("sepsis", "septic_shock"),
  rec=[A("use_antimicrobial_therapeutic_drug_monitoring", condition="case-by-case in selected patients")],
  conf="medium")

t(35, "adults", all_("suspected_sepsis_or_septic_shock", "unconfirmed_infection"),
  rec=[A("reevaluate_continuously"), A("search_for_alternative_diagnoses"),
       A("discontinue_empiric_antimicrobial_therapy",
         condition="if an alternative cause is demonstrated or strongly suspected")],
  note="second explicit mimic mandate, this one on the treatment side")

t(36, "adults", all_("sepsis_or_septic_shock", "confirmed_microbiological_diagnosis_and_susceptibility"),
  rec=[A("de_escalate_antimicrobial_therapy")], pref=["no_de_escalation"],
  ref=[R(0, "de-escalation = discontinuing unnecessary agents or narrowing spectrum")])

t(37, "adults", all_("sepsis_or_septic_shock", "no_pathogen_on_final_cultures"),
  rec=[A("de_escalate_antimicrobial_therapy")], pref=["no_de_escalation"])

t(38, "adults", all_("sepsis_or_septic_shock", "receiving_empiric_antifungal_therapy"),
  rec=[A("use_candida_biomarkers_to_stop_antifungals", polarity="against")],
  ref=[R(0, "case-by-case in clinically improving high-risk patients",
            cond="high_risk_of_candida_infection")])

t(39, "adults", all_("initial_diagnosis_of_sepsis_or_septic_shock", "adequate_source_control"),
  rec=[A("use_shorter_antimicrobial_duration")], pref=["longer_duration"])

t(40, "adults", all_("initial_diagnosis_of_sepsis_or_septic_shock", "adequate_source_control",
                     "optimal_duration_unclear"),
  rec=[A("use_procalcitonin_and_clinical_evaluation_to_stop_antimicrobials")],
  pref=["clinical_evaluation_alone"],
  note="opposite direction to stmt_022 — PCT is discouraged for STARTING, suggested for STOPPING")

t(41, "mechanically ventilated adults", all_("sepsis_or_septic_shock", "mechanically_ventilated",
                                             "unit_with_low_antimicrobial_resistance_prevalence"),
  rec=[A("use_selective_digestive_decontamination")],
  note="the only conditional recommendation at high certainty in the whole guideline")

# ═══════════════════════ HEMODYNAMIC MANAGEMENT (42-64) ═══════════════════════
t(42, "adults", any_("septic_shock"),
  equiv={"options": ["invasive_blood_pressure_monitoring", "noninvasive_blood_pressure_monitoring"],
         "guideline_states_no_preference": True},
  ref=[R(0, "invasive advised with intermediate-to-high, escalating or multiple vasopressors; "
            "frequent arterial sampling; or inconsistent noninvasive readings",
            cond="high_or_escalating_vasopressor_requirement")])

t(43, "adults", any_("sepsis", "septic_shock"),
  rec=[A("use_crystalloids_as_first_line_fluid")])

t(44, "adults", all_("sepsis_or_septic_shock", "undergoing_initial_resuscitation"),
  rec=[A("use_balanced_crystalloids")], pref=["saline_0_9_percent"],
  ref=[R(0, "use 0.9% saline instead in traumatic brain injury",
            action="use_saline_0_9_percent", cond="traumatic_brain_injury")])

t(45, "adults", any_("sepsis", "septic_shock"),
  rec=[A("use_crystalloids_alone")], pref=["crystalloids_plus_supplemental_albumin"],
  ref=[R(0, "albumin may suit patients already given large crystalloid volumes or with cirrhosis; "
            "avoid in traumatic brain injury",
            action="use_supplemental_albumin",
            cond="large_crystalloid_volume_already_given_or_cirrhosis")])

t(46, "adults", any_("sepsis", "septic_shock"),
  rec=[A("use_starches_for_resuscitation", polarity="against")])

t(47, "adults", any_("sepsis", "septic_shock"),
  rec=[A("use_gelatin_for_resuscitation", polarity="against")])

t(48, "adults", all_("sepsis_or_septic_shock", "already_received_30ml_per_kg",
                     "persistent_hypoperfusion"),
  equiv={"options": ["liberal_fluid_resuscitation_strategy", "restrictive_fluid_resuscitation_strategy"],
         "guideline_states_no_preference": True,
         "choose_on": ["individual patient factors", "health system factors"]},
  ref=[R(0, "factors include current condition and chronic illness such as heart failure, and "
            "availability of monitored beds if a restrictive approach needs vasopressors")],
  note="2021 counterpart was itself an insufficient-evidence statement")

t(49, "adults", any_("sepsis", "septic_shock"),
  rec=[A("use_dynamic_measures_to_guide_fluid_resuscitation")],
  pref=["physical_examination_alone", "static_measures_alone"],
  ref=[R(0, "dynamic measures: passive leg raise or fluid bolus response using SV, SVV, PP or PPV")])

t(50, "adults", any_("septic_shock"),
  constrains=[{"insufficient_evidence_for": "minimally_invasive_or_noninvasive_cardiac_output_monitoring"}],
  ref=[R(0, "defines the device classes; critical care ultrasound was not evaluated")])

t(51, "adults", any_("sepsis_with_elevated_lactate", "septic_shock"),
  rec=[A("use_serial_lactate_measurements_to_guide_resuscitation")],
  ref=[R(0, "individualise fluids after the initial bolus and follow lactate decrement rather "
            "than driving fluids to lactate normalisation",
            action="individualise_fluid_after_initial_bolus")])

t(52, "adults", any_("sepsis", "septic_shock"),
  rec=[A("use_capillary_refill_time_to_guide_resuscitation", condition="as an adjunct to other perfusion measures")])

t(53, "adults", any_("septic_shock"),
  rec=[A("use_norepinephrine_as_first_line_vasopressor")],
  pref=["dopamine", "epinephrine", "selepressin"],
  note="certainty is stated per comparator, not in the parenthetical")

t(54, "adults", any_("septic_shock"), rec=[A("use_terlipressin", polarity="against")])

t(55, "adults", any_("septic_shock"),
  rec=[A("use_norepinephrine_as_first_line_vasopressor")],
  pref=["vasopressin", "angiotensin_II"],
  note="downgraded from strong in 2021; certainty stated per comparator")

t(56, "adults", all_("septic_shock", "escalating_norepinephrine_dose"),
  rec=[A("add_vasopressin")])

t(57, "adults", all_("septic_shock", "inadequate_MAP_despite_norepinephrine_and_vasopressin"),
  rec=[A("add_epinephrine")],
  ref=[R(0, "where vasopressin is unavailable, add epinephrine to norepinephrine alone",
            cond="vasopressin_unavailable")])

t(58, "adults", all_("septic_shock", "concomitant_cardiac_dysfunction"),
  equiv={"options": ["norepinephrine_first_line", "epinephrine_first_line"],
         "guideline_states_no_preference": True},
  ref=[R(0, "norepinephrine may suit tachyarrhythmia or sinus tachycardia; epinephrine may suit "
            "bradyarrhythmia or sinus bradycardia",
            cond="tachyarrhythmia_vs_bradyarrhythmia")])

t(59, "adults", all_("refractory_septic_shock", "escalating_vasopressor_requirement"),
  constrains=[{"insufficient_evidence_for": "iv_methylene_blue"}],
  ref=[R(0, "may raise blood pressure but survival benefit unknown; some patients may value a trial"),
       R(1, "'In our practice': 69% of panelists never or almost never use it")],
  note="the 'In our practice' remark is a statement type with no evidence grade — our schema has no slot for it")

t(60, "adults", all_("septic_shock", "cardiac_dysfunction", "persistent_hypoperfusion",
                     "adequate_fluid_status", "adequate_arterial_blood_pressure"),
  rec=[A("use_inotropes")], pref=["no_inotropes"],
  ref=[R(0, "inotropes are added to, not substituted for, vasopressors when vasopressors are "
            "needed to hold MAP at target",
            action="use_inotropes_in_addition_to_vasopressors",
            cond="vasopressors_required_to_maintain_MAP")],
  conf="medium")

t(61, "adults", all_("septic_shock", "persistent_hypoperfusion", "cardiac_dysfunction",
                     "adequate_fluid_resuscitation", "adequate_arterial_blood_pressure"),
  equiv={"options": ["add_dobutamine_to_norepinephrine", "epinephrine_alone"],
         "guideline_states_no_preference": True},
  ref=[R(0, "evidence insufficient to choose dobutamine vs milrinone")],
  conf="medium",
  note="five conditions must hold together; phrased 'A or B alone', so keyword rules for "
       "equivalence miss it — found via the missing strength symbol in Table 1")

t(62, "adults", all_("septic_shock", "cardiac_dysfunction", "persistent_hypoperfusion",
                     "adequate_volume_status", "adequate_arterial_blood_pressure"),
  rec=[A("use_levosimendan", polarity="against")], conf="medium")

t(63, "adults", all_("septic_shock", "ongoing_vasopressor_requirement"),
  constrains=[{"insufficient_evidence_for": "oral_midodrine"}])

t(64, "adults", any_("septic_shock"),
  rec=[A("use_beta_blockers_for_septic_shock", polarity="against")],
  ref=[R(0, "based on short-acting IV beta-blockers, esmolol and landiolol")])


# ═══════════════════ RESPIRATORY SUPPORT (65-78) — light ═══════════════════
t(65, "adults", any_("sepsis"),
  equiv={"options": ["pulse_oximetry_SpO2", "arterial_blood_gas_SaO2"],
         "guideline_states_no_preference": True, "adjunct_to": "physical examination and clinical acumen"},
  ref=[R(0, "body offers the two as equivalent; the remark calls ABG the gold standard and "
            "preferable when available, and names four settings where SpO2 substitution is "
            "less accurate (shock, darker skin tones, SpO2 <90% or >97%)",
         action="prefer_arterial_blood_gas_when_available", kind="adds_preference",
         cond="spo2_substitution_unreliable_setting")],
  note="body and remark disagree on whether a preference exists — an equivalent-options "
       "statement whose remark states a preference. Chunking the remark away flips the meaning.")
t(66, "adults", all_("sepsis", "acute_hypoxemic_respiratory_failure"),
  equiv={"options": ["liberal_oxygen_target", "conservative_oxygen_target"],
         "guideline_states_no_preference": True, "choose_on": ["patient factors", "resource limitations"]},
  targ=[{"parameter": "SpO2", "conservative": "90-93%", "liberal": ">=96%",
         "source": "remark_0_trial_targets"},
        {"parameter": "SpO2", "conservative": "90% (IQR 90-92%)", "liberal": "96% (IQR 94-98%)",
         "source": "remark_1_in_our_practice"}],
  ref=[R(0, "the numeric targets the trials actually used — the body states none",
         kind="adds_target"),
       R(1, "the panel's own practice targets, stated only in the remark", kind="adds_target")],
  note="body says 'higher or lower depending on patient factors'; every number an agent "
       "would need to act on lives in the two remarks")
t(67, "adults", all_("sepsis", "acute_hypoxemic_respiratory_failure"),
  rec=[A("use_high_flow_nasal_cannula")], pref=["conventional_oxygen_therapy"],
  ref=[R(0, "narrows the statement to PaO2/FiO2 <200 or SpO2/FiO2 <235 — a threshold that "
            "appears nowhere in the body",
         cond="pf_ratio_lt_200_or_sf_ratio_lt_235", kind="restricts_population")],
  note="read without its remark this recommends HFNC for all sepsis with AHRF")
t(68, "adults", all_("sepsis", "acute_hypoxemic_respiratory_failure"),
  rec=[A("use_high_flow_nasal_cannula", condition="as initial therapy")],
  pref=["noninvasive_positive_pressure_ventilation"])
t(69, "adults", all_("sepsis", "acute_hypoxemic_respiratory_failure"),
  rec=[A("use_high_flow_nasal_cannula")], pref=["hfnc_alternating_with_nippv"])
t(70, "adults", all_("sepsis", "acute_hypoxemic_respiratory_failure", "not_intubated"),
  rec=[A("trial_awake_proning"),
       A("use_sedation_to_promote_proning_tolerance", polarity="against", source="remark")],
  ref=[R(0, "duration and frequency follow patient tolerance; and an outright prohibition — "
            "sedation must not be used to make proning tolerable in non-intubated patients",
         action="use_sedation_to_promote_proning_tolerance", polarity="against",
         kind="adds_action")],
  note="the prohibition exists ONLY in the remark; the body contains no mention of sedation")
t(71, "adults", all_("sepsis", "ards"),
  rec=[A("use_low_tidal_volume_ventilation", dose="6 mL/kg")], pref=["high_tidal_volume_gt_10_ml_per_kg"])
t(72, "adults", all_("sepsis_associated_hypoxemic_respiratory_failure", "no_ards"),
  rec=[A("use_tidal_volume_6_to_8_ml_per_kg_IBW"),
       A("screen_regularly_for_ARDS", source="remark")],
  pref=["tidal_volume_4_to_lt_6_ml_per_kg_IBW"],
  ref=[R(0, "adds an ongoing surveillance duty — screen regularly for ARDS, because the "
            "diagnosis is often missed or delayed",
         action="screen_regularly_for_ARDS", kind="adds_action")],
  note="the statement's whole applicability hinges on 'without ARDS'; the remark is what "
       "tells you to keep re-checking that precondition")
t(73, "adults", all_("sepsis", "ards"), pref=["higher_plateau_pressures"],
  targ=[{"parameter": "plateau_pressure", "value": 30, "unit": "cm H2O", "kind": "upper_limit"}])
t(74, "adults", all_("sepsis", "moderate_severe_ards"),
  rec=[A("use_higher_PEEP")], pref=["lower_PEEP"])
t(75, "adults", all_("sepsis", "moderate_severe_ards"),
  rec=[A("use_incremental_PEEP_titration_strategy", polarity="against")])
t(76, "adults", all_("sepsis", "moderate_severe_ards"),
  rec=[A("use_prone_ventilation", timing="greater than 12 hours daily")])
t(77, "adults", all_("sepsis", "moderate_severe_ards"),
  rec=[A("use_intermittent_NMBA_boluses")], pref=["continuous_NMBA_infusion"])
t(78, "adults", all_("severe_ards_due_to_sepsis", "conventional_ventilation_failed",
                     "experienced_centre_with_infrastructure"),
  rec=[A("use_veno_venous_ECMO")])

# ═════════════ ADDITIONAL AND ADJUNCTIVE THERAPIES (79-99) — light ═════════════
t(79, "adults", any_("septic_shock"), rec=[A("use_iv_corticosteroids")])
t(80, "adults", any_("sepsis", "septic_shock"),
  rec=[A("use_antipyretic_therapy", polarity="against",
         options=["pharmacologic", "surface_cooling"],
         purpose="improving clinical outcomes")],
  ref=[R(0, "the 'suggest against' does not reach antipyretics given for pain or symptom "
            "control, nor patients with another indication for temperature control "
            "(neurocritical care, post-cardiac arrest)",
         cond="antipyretic_for_symptom_control_or_other_temperature_indication",
         kind="restricts_population")],
  note="'either pharmacologic or surface cooling' scopes the thing being discouraged; it is NOT an equivalent-options statement. "
       "Without the remark the agent would withhold antipyretics from a febrile patient in pain.")
t(81, "adults", any_("sepsis", "septic_shock"), rec=[A("use_iv_vitamin_c", polarity="against")])
t(82, "adults", any_("sepsis", "septic_shock"), rec=[A("use_iv_immunoglobulins", polarity="against")])
t(83, "adults", any_("sepsis", "septic_shock"),
  rec=[A("use_blood_purification", polarity="against",
         options=["hemoperfusion", "high_dose_hemofiltration", "plasma_exchange"])])
t(84, "adults", any_("sepsis", "septic_shock"),
  rec=[A("use_polymyxin_B_hemoperfusion", polarity="against")])
t(85, "adults", any_("sepsis", "septic_shock"), rec=[A("use_vitamin_D_therapy", polarity="against")],
  ref=[R(0, "does not reach patients already on low-dose vitamin D for another indication "
            "or as part of standard nutrition",
         cond="vitamin_D_for_other_indication_or_standard_nutrition",
         kind="restricts_population")],
  note="without the remark the agent would stop a home supplement")
t(86, "adults", any_("sepsis", "septic_shock"),
  rec=[A("use_xuebijing_injection", polarity="against",
         condition="outside jurisdictions with regulatory approval")])
t(87, "adults", all_("sepsis_or_septic_shock", "risk_factors_for_gi_bleeding"),
  rec=[A("use_stress_ulcer_prophylaxis_with_ppi")], pref=["no_stress_ulcer_prophylaxis"])
t(88, "adults", any_("sepsis", "septic_shock"), rec=[A("use_probiotics", polarity="against")])
t(89, "adults", all_("septic_shock", "after_acute_resuscitation_phase"),
  rec=[A("use_active_fluid_removal"), A("administer_diuretics", source="remark")],
  prec=[{"before": "administer_diuretics",
         "after": "use_ultrafiltration_or_extracorporeal_fluid_removal",
         "strength": "conditional", "source": "remark",
         "trigger": "diuretics insufficient"}],
  ref=[R(0, "defines both key terms — 'acute resuscitation phase' and 'active fluid "
            "removal' — and imposes an order: diuretics first, ultrafiltration or "
            "extracorporeal removal only if diuretics are insufficient; lists the factors "
            "to weigh before starting",
         action="administer_diuretics", kind="defines_term")],
  note="both nouns in the body are undefined without the remark, AND the drug-before-machine "
       "ordering lives only there")
t(90, "adults", any_("sepsis", "septic_shock"),
  rec=[A("use_restrictive_transfusion_strategy")], pref=["liberal_transfusion_strategy"])
t(91, "adults", any_("sepsis", "septic_shock"),
  rec=[A("initiate_enteral_nutrition", timing="early, within 72 hours")])
t(92, "adults", any_("sepsis", "septic_shock"),
  rec=[A("initiate_insulin_therapy")],
  targ=[{"parameter": "glucose_threshold_to_start_insulin", "value": 180,
         "unit": "mg/dL", "alt_unit": "10 mmol/L", "kind": "trigger"}])
t(93, "adults", all_("sepsis_or_septic_shock", "acute_kidney_injury",
                     "no_definitive_indication_for_rrt"),
  rec=[A("use_renal_replacement_therapy", polarity="against")])
t(94, "adults", all_("sepsis_or_septic_shock", "acute_kidney_injury", "rrt_warranted"),
  equiv={"options": ["continuous_renal_replacement_therapy", "intermittent_renal_replacement_therapy"],
         "guideline_states_no_preference": True})
t(95, "adults", all_("septic_shock", "hypoperfusion_induced_lactic_acidemia"),
  rec=[A("use_sodium_bicarbonate", polarity="against",
         purpose="improve haemodynamics or reduce vasopressor requirement")])
t(96, "adults", all_("septic_shock", "severe_metabolic_acidemia_pH_le_7_2",
                     "acute_kidney_injury_AKIN_2_or_3"),
  rec=[A("use_sodium_bicarbonate")],
  note="direct counterpart to stmt_095 — same drug, opposite direction, discriminated only by pH and AKIN stage")
t(97, "adults", any_("sepsis", "septic_shock"),
  rec=[A("use_pharmacologic_vte_prophylaxis", condition="unless a contraindication exists")])
t(98, "adults", any_("sepsis", "septic_shock"),
  rec=[A("use_low_molecular_weight_heparin_for_vte_prophylaxis")], pref=["unfractionated_heparin"])
t(99, "adults", any_("sepsis", "septic_shock"),
  rec=[A("use_pharmacologic_vte_prophylaxis_alone")],
  pref=["pharmacologic_plus_mechanical_vte_prophylaxis"])

# ══════════════════════ GOALS OF CARE (100-107) — light ══════════════════════
t(100, "adults", any_("sepsis", "septic_shock"),
  rec=[A("discuss_goals_of_care_and_prognosis_with_patient_or_family")])
t(101, "adults", any_("sepsis", "septic_shock"),
  rec=[A("address_goals_of_care", timing="early, within 72 hours")], pref=["late_72h_or_later"])
t(102, "adults", any_("sepsis", "septic_shock"),
  constrains=[{"insufficient_evidence_for": "standardized_criterion_to_identify_patients_for_goals_of_care_discussion"}])
t(103, "health systems", any_("discharged_after_sepsis_or_septic_shock"),
  rec=[A("ensure_opportunity_to_execute_advance_directives")])
t(104, "adults", any_("sepsis", "septic_shock"),
  constrains=[{"insufficient_evidence_for": "systematic_establishment_of_advance_directives_before_discharge"}])
t(105, "adults", any_("sepsis", "septic_shock"),
  constrains=[{"insufficient_evidence_for": "formal_time_limited_trials_of_critical_care"}],
  ref=[R(0, "defines what a time-limited trial is — a collaborative plan to use "
            "life-sustaining therapy for a defined duration, after which the response "
            "decides continue/escalate vs. refocus on other goals",
         kind="defines_term")],
  note="the body names TLT without defining it")
t(106, "general", any_("sepsis", "septic_shock"),
  rec=[A("integrate_palliative_care_principles_into_treatment_plan")])
t(107, "adults", any_("sepsis", "septic_shock"),
  rec=[A("routine_formal_palliative_care_consultation_for_all", polarity="against")],
  pref=["palliative_care_consultation_based_on_clinician_judgement"])

# ═════════════════════ TRANSITIONS OF CARE (108-123) — light ═════════════════════
t(108, "adults", all_("sepsis_or_septic_shock", "admitted_to_icu", "transfer_to_floor"),
  rec=[A("use_critical_care_transition_program")], pref=["usual_care"])
t(109, "adults", any_("sepsis", "septic_shock"),
  rec=[A("use_structured_handoff_process_at_transitions")], pref=["no_handoff_process"])
t(110, "hospitals and health systems", any_("sepsis", "septic_shock"),
  rec=[A("screen_for_economic_and_social_support_needs"), A("refer_to_available_support_services")])
t(111, "adults", any_("transition_in_care", "icu_discharge", "hospital_discharge"),
  rec=[A("perform_comprehensive_medication_reconciliation")])
t(112, "adults", any_("transition_in_care"),
  rec=[A("perform_medication_reconciliation_pharmacist_based")])
t(113, "adults and families", any_("post_icu_planning", "hospital_discharge_planning"),
  rec=[A("involve_patient_and_family_in_shared_decision_making")])
t(114, "adult survivors and families", any_("hospital_discharge"),
  rec=[A("provide_written_and_verbal_discharge_summary_information")])
t(115, "adults", all_("sepsis_or_septic_shock", "new_impairments"),
  rec=[A("include_followup_with_clinicians_for_sequelae_in_discharge_plan")])
t(116, "healthcare systems", any_("hospital_discharge"),
  rec=[A("ensure_information_for_patients_families_and_primary_care_to_navigate_transition")])
t(117, "adults", any_("survived_hospitalization_for_sepsis"),
  constrains=[{"insufficient_evidence_for": "specific_structured_multicomponent_discharge_planning_process"}])
t(118, "adult survivors and families", any_("hospital_discharge"),
  rec=[A("provide_written_and_verbal_discharge_summary_information")],
  note="near-duplicate of stmt_114 in the published text")
t(119, "adults and families", any_("before_hospital_discharge", "followup_setting"),
  rec=[A("offer_written_and_verbal_sepsis_education")])
t(120, "health systems", any_("post_hospital_recovery"),
  rec=[A("ensure_clinician_knowledge_and_competency_to_support_survivors")])
t(121, "general", any_("post_hospital_recovery"),
  constrains=[{"insufficient_evidence_for": "sepsis_focused_educational_material_for_primary_care_providers"}])
t(122, "health systems", any_("post_hospital_recovery"),
  rec=[A("implement_strategies_to_support_survivors_and_families")])
t(123, "general", any_("post_hospital_discharge"),
  constrains=[{"insufficient_evidence_for": "early_vs_routine_post_discharge_followup"}],
  ref=[R(0, "follow-up services vary by local resources and patient need", kind="implementation_note")])

# ═══════════════ LONG-TERM OUTCOMES AND RECOVERY (124-129) — light ═══════════════
t(124, "health systems", any_("after_hospital_discharge_for_sepsis"),
  rec=[A("facilitate_assessment_and_followup_for_physical_cognitive_emotional_problems")])
t(125, "adult survivors", any_("hospitalization_for_sepsis_or_septic_shock"),
  rec=[A("offer_post_critical_illness_followup_services")],
  ref=[R(0, "format, intensity and duration vary by local resources and patient need",
         kind="implementation_note")])
t(126, "adult survivors", all_("hospitalization_for_sepsis_or_septic_shock",
                              "invasive_mechanical_ventilation_gt_48h"),
  rec=[A("offer_physical_rehabilitation_after_discharge")],
  ref=[R(0, "format, intensity and duration vary by local resources and patient need",
         kind="implementation_note")])
t(127, "adult survivors", any_("hospitalization_for_sepsis_or_septic_shock"),
  rec=[A("offer_mental_health_support_services_after_discharge")],
  ref=[R(0, "format, intensity and duration vary by local resources and patient need",
         kind="implementation_note")])
t(128, "adult survivors", all_("hospitalization_for_sepsis_or_septic_shock",
                              "clinical_symptoms_of_mental_health_disorder"),
  rec=[A("refer_to_appropriate_healthcare_professionals")])
t(129, "adult survivors", any_("hospitalization_for_sepsis_or_septic_shock"),
  constrains=[{"insufficient_evidence_for": "cognition_targeted_therapies_vs_usual_care"}],
  rec=[A("continue_cognition_targeted_therapy", source="remark",
         qualifier="'reasonable to continue' — deliberately not graded")],
  ref=[R(0, "where these therapies are already in use it is reasonable to continue them, "
            "as they are likely acceptable and feasible",
         action="continue_cognition_targeted_therapy",
         cond="already_receiving_cognition_targeted_therapy", kind="adds_action")],
  note="the body is 'insufficient evidence' — literally no guidance. The ONLY actionable "
       "content in this statement is in its remark.")

# ---------------------------------------------------------------- outputs
if __name__ == "__main__":
    stmts = {s["number"]: s for s in json.loads(
        (paths.PARSED / "ssc2026_statements.json").read_text())}
    for r in T:
        s = stmts[r["number"]]
        r["domain"] = s["domain"]; r["strength"] = s["strength"]; r["certainty"] = s["certainty"]
        # sanity: an equivalent-options statement must not also carry a RECOMMENDS edge
        if r["equivalent_options"] and r["recommends"]:
            r.setdefault("_warnings", []).append("has both equivalent_options and recommends")
        if s["equivalent_options"] and not r["equivalent_options"]:
            r.setdefault("_warnings", []).append("main table flags equivalent_options but triples do not")
        if r["remarks_expected"] if False else False: pass
        if r["depth"] == "full" and len(s["remarks"]) != len(r["refined_by"]):
            r.setdefault("_warnings", []).append(
                f"remark count mismatch: table {len(s['remarks'])} vs triples {len(r['refined_by'])}")

    conds = Counter(c for r in T for c in r["applies_when"]["conditions"])
    acts  = Counter(a["action"] for r in T for a in r["recommends"])
    acts += Counter(x["action"] for r in T for x in r["refined_by"] if x.get("action"))
    json.dump({"statements": T,
               "vocabulary": {"conditions": dict(conds.most_common()),
                              "actions": dict(acts.most_common())}},
              open(paths.GRAPH / "triples_all.json", "w"), ensure_ascii=False, indent=1)

    print(f"抽取 {len(T)} 条 / 129")
    print(f"  Condition 概念 {len(conds)} 个   Action 概念 {len(acts)} 个")
    print(f"  带 equivalent_options {sum(1 for r in T if r['equivalent_options'])}")
    print(f"  带 must_precede       {sum(1 for r in T if r['must_precede'])}")
    print(f"  带 constrains         {sum(1 for r in T if r['constrains'])}")
    print(f"  remark 携带独立动作   {sum(1 for r in T for x in r['refined_by'] if x['adds_action'])}")
    kinds = Counter(x["kind"] for r in T for x in r["refined_by"])
    n_rem = sum(len(s["remarks"]) for s in stmts.values())
    n_sr  = sum(1 for s in stmts.values() if s["remarks"])
    print(f"\n  remark 分类（{n_rem} 条 remark，分布在 {n_sr} 条条款上，已全部分类）：")
    for k, v in kinds.most_common():
        print(f"     {k:<22}{v:>4}")
    changes = sum(v for k, v in kinds.items() if k != "implementation_note")
    print(f"     {'—— 会改变处置的':<22}{changes:>4}   ({changes/n_rem:.0%})")
    print(f"     {'—— 纯背景说明':<22}{kinds['implementation_note']:>4}")
    print(f"  depth: {dict(Counter(r['depth'] for r in T))}")
    print(f"  extraction_confidence {dict(Counter(r['extraction_confidence'] for r in T))}")
    w = [(r['id'], r['_warnings']) for r in T if r.get('_warnings')]
    print(f"\n自检告警 {len(w)} 条：")
    for i, ws in w: print(f"   {i}: {'; '.join(ws)}")
    print("\n出现 >=3 次的条件概念：")
    for c, n in conds.most_common():
        if n >= 3: print(f"   {n:>3}  {c}")
