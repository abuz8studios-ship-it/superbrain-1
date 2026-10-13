# -*- coding: utf-8 -*-
"""build_packet_a10.py - SUPERBRAIN-1 Packet A10 (Arabic science word problems, train + exam).

Fully programmatic: every question, worked solution and gold answer is produced by the
templates and Python arithmetic in this file (random.Random(10)). No LLM text, no teacher outputs.

Subjects: physics (kinematics, Newton, energy, power, Ohm, density, pressure, waves),
          chemistry (molar mass from a FIXED table, moles, molarity, dilution, ideal gas, stoichiometry),
          biology/earth (Punnett monohybrid/dihybrid, growth by doubling, lapse rate, seismic, unit conversion).

Schema (same as packet_a9 / packet_b_math): task, src, prompt (HEAD + question + "\\nالحل:"),
completion (" steps\\nالجواب: <number> <unit>\\n"), answer (number string), qhash
+ extra: subject, topic, template, unit, answer_value (float), tol_rel (0.01).

Split: each generator has 3 phrasings; phrasings 0/1 -> train, phrasing 2 -> exam ONLY,
and exam parameter tuples are disjoint from all train parameter tuples of that generator.
Leakage check (exact + normalized prompt and question hash) vs: train<->exam, packet_b_math
(A9 exam) and packet_b (Packet B).
"""
import os, re, json, math, random, hashlib, collections, sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
T = os.environ.get("A10_DIR", r"E:\ABU\02_PROJECTS\AHMAD-1\train")
SEED = 10
N_TRAIN_PER_GEN = 223
N_EXAM_PER_GEN = 38
TOL_REL = 1e-2
HEAD = "المهمة: حل السؤال التالي خطوة بخطوة ثم اكتب الجواب النهائي رقمًا.\nسؤال: "


def fmt(x, sig=4):
    """Plain decimal, `sig` significant figures, no exponent, trailing zeros stripped."""
    if x == 0:
        return "0"
    d = sig - 1 - int(math.floor(math.log10(abs(x))))
    y = round(x, d)
    s = f"{y:.{max(d, 0)}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s in ("-0", ""):
        s = "0"
    return s


def ch(r, xs):
    return xs[r.randrange(len(xs))]


def rv(r, lo, hi, step):
    """random value on a grid lo..hi with given step (returned as rounded float/int)."""
    n = int(round((hi - lo) / step))
    v = round(lo + step * r.randint(0, n), 6)
    return int(round(v)) if float(v).is_integer() else v


NAMES = ["أحمد", "خالد", "يوسف", "عمر", "حسن", "سلمان", "زياد", "طارق", "كريم", "فيصل",
         "علي", "سعيد", "ماجد", "إبراهيم", "باسل", "هشام", "نادر", "وليد", "منصور", "راشد"]  # male: verbs are masculine

# ---------------------------------------------------------------- fixed periodic table (g/mol)
PT = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "Na": 22.990, "Mg": 24.305, "Al": 26.982,
      "S": 32.06, "Cl": 35.45, "K": 39.098, "Ca": 40.078, "Fe": 55.845, "Cu": 63.546, "Zn": 65.38}
PT_LINE = "، ".join(f"{k}={v}" for k, v in PT.items())
COMP = {  # formula: (arabic name, composition)
    "H2O": ("الماء", {"H": 2, "O": 1}), "CO2": ("ثاني أكسيد الكربون", {"C": 1, "O": 2}),
    "NaCl": ("كلوريد الصوديوم", {"Na": 1, "Cl": 1}), "NH3": ("الأمونيا", {"N": 1, "H": 3}),
    "CH4": ("الميثان", {"C": 1, "H": 4}), "C6H12O6": ("الغلوكوز", {"C": 6, "H": 12, "O": 6}),
    "H2SO4": ("حمض الكبريتيك", {"H": 2, "S": 1, "O": 4}), "CaCO3": ("كربونات الكالسيوم", {"Ca": 1, "C": 1, "O": 3}),
    "NaOH": ("هيدروكسيد الصوديوم", {"Na": 1, "O": 1, "H": 1}), "HCl": ("كلوريد الهيدروجين", {"H": 1, "Cl": 1}),
    "KCl": ("كلوريد البوتاسيوم", {"K": 1, "Cl": 1}), "MgO": ("أكسيد المغنيسيوم", {"Mg": 1, "O": 1}),
    "Al2O3": ("أكسيد الألومنيوم", {"Al": 2, "O": 3}), "Fe2O3": ("أكسيد الحديد الثلاثي", {"Fe": 2, "O": 3}),
    "CuSO4": ("كبريتات النحاس", {"Cu": 1, "S": 1, "O": 4}), "C2H5OH": ("الإيثانول", {"C": 2, "H": 6, "O": 1}),
    "ZnCl2": ("كلوريد الخارصين", {"Zn": 1, "Cl": 2}), "CaCl2": ("كلوريد الكالسيوم", {"Ca": 1, "Cl": 2}),
    "KNO3": ("نترات البوتاسيوم", {"K": 1, "N": 1, "O": 3}), "HNO3": ("حمض النيتريك", {"H": 1, "N": 1, "O": 3}),
    "C3H8": ("البروبان", {"C": 3, "H": 8}), "Na2CO3": ("كربونات الصوديوم", {"Na": 2, "C": 1, "O": 3}),
    "MgCl2": ("كلوريد المغنيسيوم", {"Mg": 1, "Cl": 2}), "SO2": ("ثاني أكسيد الكبريت", {"S": 1, "O": 2}),
}
SOLUTES = ["NaCl", "NaOH", "KCl", "C6H12O6", "CuSO4", "KNO3", "CaCl2", "MgCl2", "Na2CO3", "ZnCl2"]


def mm(f):
    return sum(PT[e] * n for e, n in COMP[f][1].items())


def mm_steps(f):
    parts = [f"{n}×{PT[e]}" if n > 1 else f"{PT[e]}" for e, n in COMP[f][1].items()]
    return f"الكتلة المولية لـ {f} = " + " + ".join(parts) + f" = {fmt(mm(f), 5)} غ/مول."


# ---------------------------------------------------------------- generators
# each returns (key_tuple, [phrasing0, phrasing1, phrasing2], steps, value, unit, subject, topic)

def g_kin_v(r):
    u, a, t = rv(r, 0, 30, 1), rv(r, 0.5, 6, 0.5), rv(r, 2, 20, 1)
    nm = ch(r, NAMES)
    v = u + a * t
    q = [f"تتحرك سيارة بسرعة ابتدائية {u} م/ث، ثم تتسارع بانتظام بتسارع {a} م/ث² لمدة {t} ثانية. ما سرعتها النهائية بوحدة م/ث؟",
         f"يقود {nm} دراجة نارية سرعتها {u} م/ث، وزاد سرعته بتسارع ثابت مقداره {a} م/ث² خلال {t} ثانية. احسب السرعة النهائية (م/ث).",
         f"جسم يبدأ حركته بسرعة {u} م/ث ويكتسب تسارعًا منتظمًا قدره {a} م/ث². كم تصبح سرعته بعد مرور {t} ثانية بوحدة م/ث؟"]
    s = f"نستخدم معادلة الحركة: ع = ع0 + ت × ز.\nع = {u} + {a} × {t} = {fmt(v)} م/ث."
    return (u, a, t, nm), q, s, v, "م/ث", "physics", "kinematics"


def g_kin_s(r):
    u, a, t = rv(r, 0, 20, 1), rv(r, 0.5, 5, 0.5), rv(r, 2, 15, 1)
    d = u * t + 0.5 * a * t * t
    q = [f"انطلق قطار بسرعة ابتدائية {u} م/ث وتسارع بانتظام بمقدار {a} م/ث² لمدة {t} ثانية. ما المسافة التي قطعها بالمتر؟",
         f"تتحرك كرة بسرعة {u} م/ث على أرض مستوية وتتسارع بانتظام بتسارع {a} م/ث². احسب المسافة المقطوعة بالمتر خلال {t} ثانية.",
         f"ما الإزاحة بالمتر لعربة سرعتها الابتدائية {u} م/ث وتسارعها الثابت {a} م/ث² بعد زمن قدره {t} ثانية؟"]
    s = (f"نستخدم: ف = ع0 × ز + ½ × ت × ز².\nع0 × ز = {u} × {t} = {fmt(u * t)}.\n"
         f"½ × {a} × {t}² = {fmt(0.5 * a * t * t)}.\nف = {fmt(u * t)} + {fmt(0.5 * a * t * t)} = {fmt(d)} م.")
    return (u, a, t), q, s, d, "م", "physics", "kinematics"


def g_kin_avg(r):
    dkm, tmin = rv(r, 5, 400, 5), rv(r, 10, 300, 5)
    nm = ch(r, NAMES)
    v = dkm / (tmin / 60)
    q = [f"قطع {nm} بسيارته مسافة {dkm} كم في {tmin} دقيقة. ما متوسط سرعته بوحدة كم/ساعة؟",
         f"حافلة قطعت {dkm} كيلومترًا خلال {tmin} دقيقة. احسب سرعتها المتوسطة بالكيلومتر في الساعة (كم/ساعة).",
         f"سافر {nm} مسافة {dkm} كم واستغرقت الرحلة {tmin} دقيقة. كم كانت السرعة المتوسطة للرحلة بوحدة كم/ساعة؟"]
    s = (f"نحوّل الزمن إلى ساعات: {tmin} ÷ 60 = {fmt(tmin / 60)} ساعة.\n"
         f"السرعة المتوسطة = المسافة ÷ الزمن = {dkm} ÷ {fmt(tmin / 60)} = {fmt(v)} كم/ساعة.")
    return (dkm, tmin, nm), q, s, v, "كم/ساعة", "physics", "kinematics"


def g_newton_F(r):
    m, a = rv(r, 1, 1500, 1), rv(r, 0.2, 9.8, 0.2)
    F = m * a
    q = [f"ما القوة المحصلة بالنيوتن اللازمة لإكساب جسم كتلته {m} كغ تسارعًا مقداره {a} م/ث²؟",
         f"صندوق كتلته {m} كغ يتسارع بمقدار {a} م/ث² على سطح أملس. احسب القوة المؤثرة فيه بالنيوتن.",
         f"وفق قانون نيوتن الثاني، كم نيوتن تساوي القوة المحصلة على جسم كتلته {m} كغ إذا كان تسارعه {a} م/ث²؟"]
    s = f"قانون نيوتن الثاني: ق = ك × ت.\nق = {m} × {a} = {fmt(F)} نيوتن."
    return (m, a), q, s, F, "نيوتن", "physics", "newton"


def g_newton_a(r):
    m = rv(r, 2, 200, 1)
    f = rv(r, 1, 100, 1)
    F = f + rv(r, 5, 600, 5)
    a = (F - f) / m
    q = [f"يُسحب صندوق كتلته {m} كغ بقوة أفقية مقدارها {F} نيوتن، وتؤثر فيه قوة احتكاك معاكسة مقدارها {f} نيوتن. ما تسارعه بوحدة م/ث²؟",
         f"دُفعت عربة كتلتها {m} كغ بقوة {F} نيوتن، بينما يعيقها احتكاك قدره {f} نيوتن. احسب تسارع العربة (م/ث²).",
         f"تؤثر قوة دفع مقدارها {F} نيوتن في جسم كتلته {m} كغ، وقوة الاحتكاك {f} نيوتن في الاتجاه المعاكس. أوجد التسارع الناتج بوحدة م/ث²."]
    s = (f"القوة المحصلة = {F} − {f} = {F - f} نيوتن.\n"
         f"التسارع = القوة المحصلة ÷ الكتلة = {F - f} ÷ {m} = {fmt(a)} م/ث².")
    return (m, f, F), q, s, a, "م/ث²", "physics", "newton"


def g_weight(r):
    m = rv(r, 0.5, 250, 0.5)
    g = ch(r, [9.8, 1.6, 3.7])
    body = {9.8: "الأرض", 1.6: "القمر", 3.7: "المريخ"}[g]
    W = m * g
    q = [f"ما وزن جسم كتلته {m} كغ على سطح {body} حيث تسارع الجاذبية {g} م/ث²؟ اكتب الجواب بالنيوتن.",
         f"رائد فضاء يحمل جهازًا كتلته {m} كغ على سطح {body} (تسارع الجاذبية {g} م/ث²). احسب وزن الجهاز بالنيوتن.",
         f"احسب بالنيوتن قوة الجاذبية المؤثرة في كتلة مقدارها {m} كغ موضوعة على {body}، علمًا بأن الجاذبية هناك {g} م/ث²."]
    s = f"الوزن = الكتلة × تسارع الجاذبية.\nو = {m} × {g} = {fmt(W)} نيوتن."
    return (m, g), q, s, W, "نيوتن", "physics", "newton"


def g_ke(r):
    m, v = rv(r, 0.1, 2000, 0.1), rv(r, 1, 60, 1)
    KE = 0.5 * m * v * v
    q = [f"ما الطاقة الحركية بالجول لجسم كتلته {m} كغ يتحرك بسرعة {v} م/ث؟",
         f"تتحرك كرة كتلتها {m} كغ بسرعة {v} م/ث. احسب طاقتها الحركية بوحدة الجول.",
         f"أوجد بالجول مقدار الطاقة الحركية التي يمتلكها جسم كتلته {m} كغ عندما تبلغ سرعته {v} م/ث."]
    s = f"الطاقة الحركية = ½ × ك × ع².\nط = ½ × {m} × {v}² = ½ × {m} × {v * v} = {fmt(KE)} جول."
    return (m, v), q, s, KE, "جول", "physics", "energy"


def g_pe(r):
    m, h = rv(r, 0.5, 500, 0.5), rv(r, 1, 120, 1)
    g = 9.8
    PE = m * g * h
    q = [f"رُفع جسم كتلته {m} كغ إلى ارتفاع {h} م. ما طاقة الوضع المكتسبة بالجول؟ (تسارع الجاذبية 9.8 م/ث²)",
         f"يقف صندوق كتلته {m} كغ على رف ارتفاعه {h} م عن الأرض. احسب طاقة الوضع الثقالية له بالجول، باعتبار الجاذبية 9.8 م/ث².",
         f"كم جولًا من طاقة الوضع يختزن جسم كتلته {m} كغ موضوع على ارتفاع {h} م فوق سطح الأرض؟ استخدم ج = 9.8 م/ث²."]
    s = f"طاقة الوضع = ك × ج × ع.\nط = {m} × 9.8 × {h} = {fmt(PE)} جول."
    return (m, h), q, s, PE, "جول", "physics", "energy"


def g_power(r):
    F, d, t = rv(r, 10, 2000, 10), rv(r, 1, 100, 1), rv(r, 2, 120, 1)
    nm = ch(r, NAMES)
    W = F * d
    P = W / t
    q = [f"دفع {nm} صندوقًا بقوة {F} نيوتن مسافة {d} م خلال {t} ثانية. ما القدرة المبذولة بالواط؟",
         f"رافعة ترفع حملًا بقوة {F} نيوتن مسافة {d} م في زمن {t} ثانية. احسب قدرتها بالواط.",
         f"بذل عامل شغلًا بقوة ثابتة مقدارها {F} نيوتن لتحريك جسم {d} م في اتجاه القوة، واستغرق ذلك {t} ثانية. أوجد القدرة بوحدة الواط."]
    s = f"الشغل = ق × ف = {F} × {d} = {W} جول.\nالقدرة = الشغل ÷ الزمن = {W} ÷ {t} = {fmt(P)} واط."
    return (F, d, t, nm), q, s, P, "واط", "physics", "energy"


def g_ohm_I(r):
    V, R = rv(r, 1.5, 240, 0.5), rv(r, 2, 500, 1)
    I = V / R
    q = [f"مقاومة قيمتها {R} أوم موصولة بفرق جهد {V} فولت. ما شدة التيار المار فيها بالأمبير؟",
         f"يمر تيار في سلك مقاومته {R} أوم عند تطبيق جهد مقداره {V} فولت عليه. احسب شدة التيار بوحدة الأمبير.",
         f"باستخدام قانون أوم، أوجد التيار بالأمبير في دائرة جهدها {V} فولت ومقاومتها الكلية {R} أوم."]
    s = f"قانون أوم: ت = ج ÷ م.\nت = {V} ÷ {R} = {fmt(I)} أمبير."
    return (V, R), q, s, I, "أمبير", "physics", "electricity"


def g_resist(r):
    R1, R2 = rv(r, 2, 300, 1), rv(r, 2, 300, 1)
    V = rv(r, 3, 120, 1.5)
    mode = ch(r, ["series", "parallel"])
    if mode == "series":
        Re = R1 + R2
        how = f"على التوالي: م = {R1} + {R2} = {Re} أوم."
        w = "على التوالي"
    else:
        Re = R1 * R2 / (R1 + R2)
        how = f"على التوازي: م = ({R1} × {R2}) ÷ ({R1} + {R2}) = {fmt(Re)} أوم."
        w = "على التوازي"
    I = V / Re
    q = [f"وُصلت مقاومتان {R1} أوم و{R2} أوم {w} مع بطارية جهدها {V} فولت. ما التيار الكلي المسحوب من البطارية بالأمبير؟",
         f"دائرة كهربائية فيها مقاومتان قيمتاهما {R1} أوم و{R2} أوم موصولتان {w}، والجهد الكلي {V} فولت. احسب التيار الكلي بالأمبير.",
         f"مصدر جهده {V} فولت يغذي مقاومتين ({R1} أوم و{R2} أوم) متصلتين {w}. أوجد شدة التيار الكلي في الدائرة بوحدة الأمبير."]
    s = f"المقاومة المكافئة {how}\nالتيار الكلي = {V} ÷ {fmt(Re)} = {fmt(I)} أمبير."
    return (R1, R2, V, mode), q, s, I, "أمبير", "physics", "electricity"


def g_epower(r):
    V = ch(r, [12, 110, 120, 220, 230, 240])
    I = rv(r, 0.1, 15, 0.1)
    hrs = rv(r, 1, 24, 1)
    P = V * I
    E = P * hrs / 1000
    q = [f"جهاز كهربائي يعمل على جهد {V} فولت ويسحب تيارًا مقداره {I} أمبير، وشُغّل لمدة {hrs} ساعة. ما الطاقة المستهلكة بوحدة كيلوواط·ساعة؟",
         f"مدفأة تعمل على {V} فولت وتسحب {I} أمبير. إذا بقيت تعمل {hrs} ساعة، فاحسب الطاقة الكهربائية المستهلكة بالكيلوواط·ساعة.",
         f"احسب بالكيلوواط·ساعة استهلاك جهاز يتصل بمصدر جهده {V} فولت ويمر فيه تيار {I} أمبير إذا عمل دون انقطاع مدة {hrs} ساعة."]
    s = (f"القدرة = ج × ت = {V} × {I} = {fmt(P)} واط.\n"
         f"الطاقة = القدرة × الزمن ÷ 1000 = {fmt(P)} × {hrs} ÷ 1000 = {fmt(E)} كيلوواط·ساعة.")
    return (V, I, hrs), q, s, E, "كيلوواط·ساعة", "physics", "electricity"


MATERIALS = ["قطعة معدنية", "حجر", "قطعة خشب", "سائل", "قطعة بلاستيك", "سبيكة", "عينة صخرية", "قطعة زجاج"]


def g_density(r):
    mat = ch(r, MATERIALS)
    mode = ch(r, ["rho", "mass"])
    V = rv(r, 2, 900, 1)
    if mode == "rho":
        m = rv(r, 1, 5000, 1)
        rho = m / V
        q = [f"{mat} كتلتها {m} غ وحجمها {V} سم³. ما كثافتها بوحدة غ/سم³؟",
             f"قيس حجم {mat} فكان {V} سم³ وكتلتها {m} غ. احسب كثافتها (غ/سم³).",
             f"أوجد كثافة {mat} بوحدة غ/سم³ إذا علمت أن كتلتها {m} غ وأنها تشغل حجمًا قدره {V} سم³."]
        s = f"الكثافة = الكتلة ÷ الحجم = {m} ÷ {V} = {fmt(rho)} غ/سم³."
        return (mat, mode, V, m), q, s, rho, "غ/سم³", "physics", "density"
    rho = rv(r, 0.5, 19.3, 0.1)
    m = rho * V
    q = [f"{mat} كثافتها {rho} غ/سم³ وحجمها {V} سم³. ما كتلتها بالغرام؟",
         f"إذا كانت كثافة {mat} {rho} غ/سم³ وحجمها {V} سم³، فاحسب كتلتها بوحدة الغرام.",
         f"احسب بالغرام كتلة {mat} تشغل حجمًا مقداره {V} سم³ وكثافة مادتها {rho} غ/سم³."]
    s = f"الكتلة = الكثافة × الحجم = {rho} × {V} = {fmt(m)} غ."
    return (mat, mode, V, rho), q, s, m, "غ", "physics", "density"


def g_pressure(r):
    mode = ch(r, ["FA", "hydro"])
    if mode == "FA":
        F, A = rv(r, 10, 5000, 10), rv(r, 0.01, 4, 0.01)
        P = F / A
        q = [f"تؤثر قوة مقدارها {F} نيوتن عموديًا على سطح مساحته {A} م². ما الضغط بالباسكال؟",
             f"صندوق وزنه {F} نيوتن يستقر على قاعدة مساحتها {A} م². احسب الضغط الذي يولده على الأرض بالباسكال.",
             f"أوجد بالباسكال الضغط الناتج عن قوة عمودية {F} نيوتن موزعة بانتظام على مساحة {A} م²."]
        s = f"الضغط = القوة ÷ المساحة = {F} ÷ {A} = {fmt(P)} باسكال."
        return (mode, F, A), q, s, P, "باسكال", "physics", "pressure"
    rho = ch(r, [1000, 1025, 800, 13600, 900])
    h = rv(r, 0.1, 50, 0.1)
    P = rho * 9.8 * h
    q = [f"ما الضغط الناتج عن سائل كثافته {rho} كغ/م³ عند عمق {h} م بالباسكال؟ (ج = 9.8 م/ث²، أهمل الضغط الجوي)",
         f"غواص على عمق {h} م داخل سائل كثافته {rho} كغ/م³. احسب ضغط السائل عليه بالباسكال، علمًا أن ج = 9.8 م/ث² ودون احتساب الضغط الجوي.",
         f"احسب بالباسكال ضغط عمود سائل ارتفاعه {h} م وكثافته {rho} كغ/م³، باستخدام ج = 9.8 م/ث² ومع إهمال الضغط الجوي."]
    s = f"ضغط السائل = ث × ج × ع = {rho} × 9.8 × {h} = {fmt(P)} باسكال."
    return (mode, rho, h), q, s, P, "باسكال", "physics", "pressure"


def g_wave(r):
    mode = ch(r, ["lam", "f", "v"])
    if mode == "lam":
        v, f = rv(r, 300, 1500, 10), rv(r, 20, 5000, 10)
        lam = v / f
        q = [f"موجة صوتية سرعتها {v} م/ث وترددها {f} هرتز. ما طولها الموجي بالمتر؟",
             f"ينتشر صوت بسرعة {v} م/ث بتردد {f} هرتز. احسب الطول الموجي بالمتر.",
             f"أوجد بالمتر طول موجة ترددها {f} هرتز تنتشر في وسط بسرعة {v} م/ث."]
        s = f"الطول الموجي = السرعة ÷ التردد = {v} ÷ {f} = {fmt(lam)} م."
        return (mode, v, f), q, s, lam, "م", "physics", "waves"
    if mode == "f":
        v, lam = rv(r, 300, 1500, 10), rv(r, 0.05, 10, 0.05)
        f = v / lam
        q = [f"موجة سرعتها {v} م/ث وطولها الموجي {lam} م. ما ترددها بالهرتز؟",
             f"تنتشر موجة طولها {lam} م بسرعة {v} م/ث. احسب ترددها بوحدة الهرتز.",
             f"أوجد بالهرتز تردد موجة طولها الموجي {lam} م وسرعة انتشارها {v} م/ث."]
        s = f"التردد = السرعة ÷ الطول الموجي = {v} ÷ {lam} = {fmt(f)} هرتز."
        return (mode, v, lam), q, s, f, "هرتز", "physics", "waves"
    f, lam = rv(r, 1, 1000, 1), rv(r, 0.1, 20, 0.1)
    v = f * lam
    q = [f"موجة ترددها {f} هرتز وطولها الموجي {lam} م. ما سرعة انتشارها بوحدة م/ث؟",
         f"على سطح الماء موجات ترددها {f} هرتز وطولها {lam} م. احسب سرعتها (م/ث).",
         f"أوجد سرعة موجة بوحدة م/ث إذا كان ترددها {f} هرتز وطولها الموجي {lam} م."]
    s = f"السرعة = التردد × الطول الموجي = {f} × {lam} = {fmt(v)} م/ث."
    return (mode, f, lam), q, s, v, "م/ث", "physics", "waves"


# ---------------------------------------------------------------- chemistry
def g_molar_mass(r):
    f = ch(r, sorted(COMP))
    k = 1 if r.random() < 0.15 else rv(r, 0.2, 25, 0.1)  # 1 -> molar mass; else mass of k mol
    M = mm(f) * k
    nm, _ = COMP[f]
    if k == 1:
        q = [f"استخدم الكتل الذرية التالية ({PT_LINE}). ما الكتلة المولية لمركب {nm} ({f}) بوحدة غ/مول؟",
             f"بالاعتماد على الكتل الذرية ({PT_LINE})، احسب الكتلة المولية لـ {f} ({nm}) بالغرام لكل مول.",
             f"أوجد الكتلة المولية لـ {nm} الذي صيغته {f} بوحدة غ/مول، مستعينًا بالكتل الذرية: {PT_LINE}."]
        s = mm_steps(f)
    else:
        q = [f"استخدم الكتل الذرية التالية ({PT_LINE}). ما كتلة {k} مول من {nm} ({f}) بالغرام؟",
             f"بالاعتماد على الكتل الذرية ({PT_LINE})، احسب بالغرام كتلة {k} مولات من {f} ({nm}).",
             f"أوجد بالغرام كتلة عينة من {nm} ({f}) تحتوي على {k} مول، مستعينًا بالكتل الذرية: {PT_LINE}."]
        s = mm_steps(f) + f"\nالكتلة = عدد المولات × الكتلة المولية = {k} × {fmt(mm(f), 5)} = {fmt(M, 5)} غ."
    return (f, k), q, s, M, ("غ/مول" if k == 1 else "غ"), "chemistry", "molar_mass"


def g_moles(r):
    f = ch(r, sorted(COMP))
    m = rv(r, 1, 900, 1)
    n = m / mm(f)
    nm, _ = COMP[f]
    q = [f"كم مولًا توجد في {m} غ من {nm} ({f})؟ استخدم الكتل الذرية: {PT_LINE}.",
         f"عينة من {f} ({nm}) كتلتها {m} غ. احسب عدد مولاتها، علمًا بأن الكتل الذرية هي: {PT_LINE}.",
         f"مستعينًا بالكتل الذرية ({PT_LINE})، أوجد عدد المولات في {m} غرامًا من {nm} الذي صيغته {f}."]
    s = mm_steps(f) + f"\nعدد المولات = الكتلة ÷ الكتلة المولية = {m} ÷ {fmt(mm(f), 5)} = {fmt(n)} مول."
    return (f, m), q, s, n, "مول", "chemistry", "moles"


def g_molarity(r):
    f = ch(r, SOLUTES)
    m = rv(r, 0.5, 200, 0.5)
    vml = rv(r, 50, 2000, 50)
    n = m / mm(f)
    M = n / (vml / 1000)
    nm, _ = COMP[f]
    q = [f"أُذيب {m} غ من {nm} ({f}) في ماء حتى أصبح حجم المحلول {vml} مل. ما مولارية المحلول بوحدة مول/لتر؟ الكتل الذرية: {PT_LINE}.",
         f"حُضّر محلول بإذابة {m} غ من {f} ({nm}) وإكمال الحجم إلى {vml} مل. احسب تركيزه المولاري (مول/لتر). استخدم: {PT_LINE}.",
         f"احسب التركيز المولاري بوحدة مول/لتر لمحلول حجمه {vml} مل يحتوي على {m} غ مذابة من {nm} ({f})، علمًا بأن الكتل الذرية: {PT_LINE}."]
    s = (mm_steps(f) + f"\nعدد المولات = {m} ÷ {fmt(mm(f), 5)} = {fmt(n)} مول.\n"
         f"الحجم باللتر = {vml} ÷ 1000 = {fmt(vml / 1000)} لتر.\n"
         f"المولارية = {fmt(n)} ÷ {fmt(vml / 1000)} = {fmt(M)} مول/لتر.")
    return (f, m, vml), q, s, M, "مول/لتر", "chemistry", "molarity"


def g_dilution(r):
    C1 = rv(r, 0.5, 18, 0.5)
    V1 = rv(r, 5, 500, 5)
    V2 = V1 + rv(r, 10, 2000, 10)
    C2 = C1 * V1 / V2
    f = ch(r, ["HCl", "NaOH", "H2SO4", "HNO3", "NaCl", "KCl"])
    nm, _ = COMP[f]
    q = [f"خُفف {V1} مل من محلول {nm} تركيزه {C1} مول/لتر بإضافة الماء حتى أصبح الحجم الكلي {V2} مل. ما التركيز الجديد بوحدة مول/لتر؟",
         f"أخذ طالب {V1} مل من محلول {f} تركيزه {C1} مول/لتر وأكمل حجمه بالماء إلى {V2} مل. احسب تركيز المحلول المخفف (مول/لتر).",
         f"محلول من {nm} حجمه {V1} مل وتركيزه {C1} مول/لتر، خُفف إلى حجم نهائي قدره {V2} مل. أوجد التركيز النهائي بوحدة مول/لتر."]
    s = (f"عند التخفيف: ت1 × ح1 = ت2 × ح2.\nت2 = ({C1} × {V1}) ÷ {V2} = {fmt(C1 * V1)} ÷ {V2} = {fmt(C2)} مول/لتر.")
    return (f, C1, V1, V2), q, s, C2, "مول/لتر", "chemistry", "molarity"


def g_gas(r):
    mode = ch(r, ["P", "V"])
    n = rv(r, 0.1, 10, 0.1)
    tc = rv(r, -20, 300, 1)
    TK = tc + 273
    Rg = 0.0821
    if mode == "P":
        V = rv(r, 1, 100, 1)
        P = n * Rg * TK / V
        q = [f"وُضع {n} مول من غاز مثالي في وعاء حجمه {V} لتر عند درجة حرارة {tc} درجة مئوية. ما ضغط الغاز بوحدة الضغط الجوي؟ (ثابت الغازات 0.0821 لتر·ضغط جوي/(مول·كلفن)، ك = م + 273)",
             f"غاز مثالي كميته {n} مول يشغل {V} لتر عند {tc} درجة مئوية. احسب ضغطه بالضغط الجوي باستخدام ثابت الغازات 0.0821 لتر·ضغط جوي/(مول·كلفن) وتحويل الحرارة بإضافة 273.",
             f"باستخدام قانون الغاز المثالي وثابت الغازات 0.0821 لتر·ضغط جوي/(مول·كلفن)، أوجد بوحدة الضغط الجوي ضغط {n} مول من غاز محصور في {V} لتر عند {tc} درجة مئوية (أضف 273 للتحويل إلى كلفن)."]
        s = (f"نحوّل الحرارة: {tc} + 273 = {TK} كلفن.\nض = ن × ر × د ÷ ح = {n} × 0.0821 × {TK} ÷ {V} = {fmt(P)} ضغط جوي.")
        return (mode, n, tc, V), q, s, P, "ضغط جوي", "chemistry", "ideal_gas"
    P = rv(r, 0.2, 10, 0.1)
    V = n * Rg * TK / P
    q = [f"ما الحجم باللتر الذي يشغله {n} مول من غاز مثالي عند ضغط {P} ضغط جوي ودرجة حرارة {tc} درجة مئوية؟ (ثابت الغازات 0.0821 لتر·ضغط جوي/(مول·كلفن)، ك = م + 273)",
         f"احسب حجم {n} مول من غاز مثالي باللتر تحت ضغط {P} ضغط جوي وعند {tc} درجة مئوية، مستخدمًا ثابت الغازات 0.0821 لتر·ضغط جوي/(مول·كلفن) وإضافة 273 للتحويل إلى كلفن.",
         f"باستخدام قانون الغاز المثالي وثابت الغازات 0.0821 لتر·ضغط جوي/(مول·كلفن)، أوجد باللتر حجم {n} مول من غاز عند ضغط {P} ضغط جوي وحرارة {tc} درجة مئوية (أضف 273 للتحويل إلى كلفن)."]
    s = (f"نحوّل الحرارة: {tc} + 273 = {TK} كلفن.\nح = ن × ر × د ÷ ض = {n} × 0.0821 × {TK} ÷ {P} = {fmt(V)} لتر.")
    return (mode, n, tc, P), q, s, V, "لتر", "chemistry", "ideal_gas"


REACTIONS = [  # (equation, reactant formula, coef, product formula, coef)
    ("2H2 + O2 → 2H2O", "O2", 1, "H2O", 2),
    ("CH4 + 2O2 → CO2 + 2H2O", "CH4", 1, "CO2", 1),
    ("CH4 + 2O2 → CO2 + 2H2O", "CH4", 1, "H2O", 2),
    ("N2 + 3H2 → 2NH3", "N2", 1, "NH3", 2),
    ("CaCO3 → CaO + CO2", "CaCO3", 1, "CO2", 1),
    ("2Na + Cl2 → 2NaCl", "Na", 2, "NaCl", 2),
    ("2Mg + O2 → 2MgO", "Mg", 2, "MgO", 2),
    ("C3H8 + 5O2 → 3CO2 + 4H2O", "C3H8", 1, "CO2", 3),
    ("C3H8 + 5O2 → 3CO2 + 4H2O", "C3H8", 1, "H2O", 4),
    ("4Al + 3O2 → 2Al2O3", "Al", 4, "Al2O3", 2),
    ("Zn + 2HCl → ZnCl2 + H2", "Zn", 1, "ZnCl2", 1),
    ("NaOH + HCl → NaCl + H2O", "NaOH", 1, "NaCl", 1),
    ("C6H12O6 + 6O2 → 6CO2 + 6H2O", "C6H12O6", 1, "CO2", 6),
    ("2SO2 + O2 → 2SO3", "SO2", 2, "SO3", 2),
]
EXTRA = {"H2": {"H": 2}, "O2": {"O": 2}, "N2": {"N": 2}, "Cl2": {"Cl": 2}, "Na": {"Na": 1}, "Mg": {"Mg": 1},
         "Al": {"Al": 1}, "Zn": {"Zn": 1}, "SO3": {"S": 1, "O": 3}, "CaO": {"Ca": 1, "O": 1}}


def mmx(f):
    c = COMP[f][1] if f in COMP else EXTRA[f]
    return sum(PT[e] * n for e, n in c.items())


def mmx_steps(f):
    c = COMP[f][1] if f in COMP else EXTRA[f]
    parts = [f"{n}×{PT[e]}" if n > 1 else f"{PT[e]}" for e, n in c.items()]
    return f"الكتلة المولية لـ {f} = " + " + ".join(parts) + f" = {fmt(mmx(f), 5)} غ/مول."


def g_stoich(r):
    eq, A, a, B, b = ch(r, REACTIONS)
    m = rv(r, 1, 500, 1)
    nA = m / mmx(A)
    nB = nA * b / a
    mB = nB * mmx(B)
    q = [f"في التفاعل الموزون {eq}، كم غرامًا من {B} تنتج عن تفاعل {m} غ من {A} تفاعلًا تامًا؟ الكتل الذرية: {PT_LINE}.",
         f"وفق المعادلة {eq}، تفاعل {m} غ من {A} كليًا مع كمية وافرة من المواد الأخرى. احسب كتلة {B} الناتجة بالغرام. استخدم الكتل الذرية: {PT_LINE}.",
         f"احسب بالغرام كتلة {B} المتكونة عند استهلاك {m} غ من {A} بالكامل حسب التفاعل {eq}، علمًا بأن الكتل الذرية: {PT_LINE}."]
    s = (mmx_steps(A) + "\n" + mmx_steps(B) + f"\nمولات {A} = {m} ÷ {fmt(mmx(A), 5)} = {fmt(nA)} مول.\n"
         f"نسبة المولات {B}:{A} = {b}:{a}، إذن مولات {B} = {fmt(nA)} × {b} ÷ {a} = {fmt(nB)} مول.\n"
         f"كتلة {B} = {fmt(nB)} × {fmt(mmx(B), 5)} = {fmt(mB)} غ.")
    return (eq, A, B, m), q, s, mB, "غ", "chemistry", "stoichiometry"


# ---------------------------------------------------------------- biology / earth
TRAITS = [("نبات البازلاء", "A", "لون الأزهار البنفسجي", "لون الأزهار الأبيض"),
          ("ذبابة الفاكهة", "B", "الأجنحة الطبيعية", "الأجنحة الضامرة"),
          ("الفئران", "C", "الفرو الأسود", "الفرو البني"),
          ("نبات الطماطم", "D", "الساق الطويلة", "الساق القصيرة"),
          ("الأرانب", "E", "الفرو الرمادي", "الفرو الأبيض"),
          ("نبات الذرة", "G", "الحبوب الصفراء", "الحبوب البيضاء")]
GENO = ["AA", "Aa", "aa"]


def gt(L, g):
    return g.replace("A", L.upper()).replace("a", L.lower())


def p_dom(g1, g2):
    pa = {"AA": 0.0, "Aa": 0.5, "aa": 1.0}
    rec = pa[g1] * pa[g2]
    return 1 - rec, rec


def g_punnett(r):
    org, L, dom, rec = ch(r, TRAITS)
    g1, g2 = ch(r, ["Aa", "Aa", "AA", "aa"]), ch(r, ["Aa", "aa", "Aa", "AA"])
    if g1 == "aa" and g2 == "aa":
        g2 = "Aa"
    if g1 == "AA" and g2 == "AA":
        g2 = "Aa"
    N = rv(r, 40, 2000, 4)
    want = ch(r, ["dom", "rec"])
    pd_, pr = p_dom(g1, g2)
    if (want == "rec" and pr == 0) or (want == "dom" and pd_ == 0):
        want = "dom" if want == "rec" else "rec"
    p = pd_ if want == "dom" else pr
    trait = dom if want == "dom" else rec
    G1, G2 = gt(L, g1), gt(L, g2)
    val = N * p
    q = [f"في {org}، صفة {dom} سائدة (الأليل {L.upper()}) على صفة {rec} (الأليل {L.lower()}). تزاوج فردان طرازاهما الجينيان {G1} و{G2} وأنتجا {N} فردًا. كم فردًا يُتوقع أن يحمل صفة {trait}؟",
         f"صفة {dom} في {org} يتحكم بها أليل سائد {L.upper()}، وصفة {rec} أليل متنحٍ {L.lower()}. من تلقيح {G1} × {G2} نتج {N} فردًا. احسب العدد المتوقع للأفراد ذوي صفة {trait} باستخدام مربع بانيت.",
         f"باستخدام مربع بانيت، إذا لُقح فرد طرازه {G1} مع فرد طرازه {G2} من {org}، حيث {L.upper()} لصفة {dom} (سائدة) و{L.lower()} لصفة {rec} (متنحية)، فما العدد المتوقع لذوي صفة {trait} من أصل {N} فردًا؟"]
    s = (f"مربع بانيت للتلقيح {G1} × {G2}: احتمال الصفة المتنحية = {fmt(pr)}، واحتمال الصفة السائدة = {fmt(pd_)}.\n"
         f"احتمال صفة {trait} = {fmt(p)}.\nالعدد المتوقع = {N} × {fmt(p)} = {fmt(val)} فردًا.")
    return (org, g1, g2, N, want), q, s, val, "فردًا", "biology", "genetics"


def g_dihybrid(r):
    N = rv(r, 32, 3200, 16)
    org = ch(r, ["نبات البازلاء", "نبات الطماطم", "نبات الذرة", "ذبابة الفاكهة"])
    cls = ch(r, [("السائدتين معًا", 9), ("السائدة الأولى والمتنحية الثانية", 3),
                 ("المتنحية الأولى والسائدة الثانية", 3), ("المتنحيتين معًا", 1)])
    name, k = cls
    val = N * k / 16
    q = [f"في {org}، تزاوج فردان طراز كل منهما AaBb (صفتان مستقلتان بسيادة تامة). من بين {N} فردًا ناتجًا، كم فردًا يُتوقع أن يظهر الصفتين {name}؟",
         f"تلقيح ثنائي الهجين AaBb × AaBb في {org} أنتج {N} فردًا. احسب العدد المتوقع للأفراد الذين يظهرون الصفتين {name}، مع افتراض التوزيع المستقل والسيادة التامة.",
         f"باستخدام نسبة مندل في التلقيح الثنائي AaBb × AaBb لدى {org}، أوجد العدد المتوقع لذوي الصفتين {name} إذا كان عدد الأفراد الناتجة {N}."]
    s = (f"نسب التلقيح الثنائي AaBb × AaBb هي 9:3:3:1 من 16.\nنصيب الصفتين {name} = {k}/16.\n"
         f"العدد المتوقع = {N} × {k} ÷ 16 = {fmt(val)} فردًا.")
    return (org, name, N), q, s, val, "فردًا", "biology", "genetics"


def g_growth(r):
    N0 = rv(r, 10, 5000, 10)
    T = ch(r, [15, 20, 25, 30, 40, 45, 60])
    k = rv(r, 1, 10, 1)
    t = T * k
    N = N0 * 2 ** k
    org = ch(r, ["بكتيريا", "خميرة", "خلايا بكتيرية", "طحالب وحيدة الخلية"])
    q = [f"مزرعة {org} تبدأ بـ {N0} خلية وتتضاعف كل {T} دقيقة. كم خلية تصبح بعد {t} دقيقة؟",
         f"يتضاعف عدد خلايا {org} في مزرعة مرة كل {T} دقيقة. إذا كان العدد الابتدائي {N0} خلية، فاحسب العدد بعد مرور {t} دقيقة.",
         f"بدأت تجربة بـ {N0} خلية من {org} زمن تضاعفها {T} دقيقة. ما عدد الخلايا المتوقع عند نهاية {t} دقيقة إذا استمر التضاعف دون عوائق؟"]
    s = (f"عدد مرات التضاعف = {t} ÷ {T} = {k}.\nالعدد النهائي = {N0} × 2^{k} = {N0} × {2 ** k} = {N} خلية.")
    return (N0, T, k, org), q, s, float(N), "خلية", "biology", "growth"


def g_lapse(r):
    T0 = rv(r, -5, 45, 1)
    hm = rv(r, 100, 8800, 50)
    T = T0 - 6.5 * hm / 1000
    nm = ch(r, NAMES)
    q = [f"درجة الحرارة عند مستوى سطح البحر {T0} درجة مئوية، وتنخفض بمعدل 6.5 درجة مئوية لكل كيلومتر ارتفاع. ما درجة الحرارة على ارتفاع {hm} م بالدرجة المئوية؟",
         f"صعد {nm} جبلًا من مستوى سطح البحر حيث الحرارة {T0} درجة مئوية إلى ارتفاع {hm} م. إذا كانت الحرارة تنقص 6.5 درجة مئوية لكل 1000 م، فاحسب درجة الحرارة عند القمة.",
         f"باعتبار معدل انخفاض الحرارة مع الارتفاع 6.5 درجة مئوية لكل كيلومتر، وأن الحرارة عند سطح البحر {T0} درجة مئوية، أوجد درجة الحرارة المتوقعة على ارتفاع {hm} م."]
    s = (f"الارتفاع بالكيلومتر = {hm} ÷ 1000 = {fmt(hm / 1000)} كم.\nالانخفاض = 6.5 × {fmt(hm / 1000)} = {fmt(6.5 * hm / 1000)} درجة.\n"
         f"الحرارة = {T0} − {fmt(6.5 * hm / 1000)} = {fmt(T)} درجة مئوية.")
    return (T0, hm, nm), q, s, T, "درجة مئوية", "earth", "atmosphere"


def g_seismic(r):
    v = rv(r, 2, 8, 0.5)
    t = rv(r, 5, 300, 1)
    d = v * t
    q = [f"تنتشر موجة زلزالية بسرعة {v} كم/ث، ووصلت إلى محطة رصد بعد {t} ثانية من حدوث الزلزال. ما بعد المحطة عن مركز الزلزال بالكيلومتر؟",
         f"سجلت محطة رصد زلزالي وصول موجة سرعتها {v} كم/ث بعد {t} ثانية من وقوع الهزة. احسب المسافة بين المحطة والبؤرة بالكيلومتر.",
         f"أوجد بالكيلومتر المسافة التي قطعتها موجة زلزالية سرعتها {v} كم/ث خلال {t} ثانية."]
    s = f"المسافة = السرعة × الزمن = {v} × {t} = {fmt(d)} كم."
    return (v, t), q, s, d, "كم", "earth", "seismology"


CONV = [  # (from-unit phrase, to-unit phrase, factor or func-kind, unit label, lo, hi, step)
    ("كيلومتر", "متر", 1000, "م", 0.1, 500, 0.1),
    ("ساعة", "ثانية", 3600, "ثانية", 0.5, 48, 0.5),
    ("لتر", "مليلتر", 1000, "مل", 0.05, 50, 0.05),
    ("كيلوغرام", "غرام", 1000, "غ", 0.01, 200, 0.01),
    ("كم/ساعة", "م/ث", 1 / 3.6, "م/ث", 1, 400, 1),
    ("متر مكعب", "لتر", 1000, "لتر", 0.01, 30, 0.01),
    ("سنتيمتر", "متر", 0.01, "م", 1, 5000, 1),
    ("دقيقة", "ساعة", 1 / 60, "ساعة", 1, 1500, 1),
    ("درجة مئوية", "كلفن", "C2K", "كلفن", -80, 500, 1),
    ("درجة مئوية", "درجة فهرنهايت", "C2F", "درجة فهرنهايت", -40, 300, 1),
    ("مليغرام", "غرام", 0.001, "غ", 10, 90000, 10),
    ("ميكرومتر", "مليمتر", 0.001, "مم", 5, 90000, 5),
]


def g_convert(r):
    fu, tu, fac, ul, lo, hi, st = ch(r, CONV)
    x = rv(r, lo, hi, st)
    if fac == "C2K":
        y = x + 273.15
        how = f"ك = م + 273.15 = {x} + 273.15 = {fmt(y, 6)} كلفن."
    elif fac == "C2F":
        y = x * 9 / 5 + 32
        how = f"ف = م × 9 ÷ 5 + 32 = {x} × 9 ÷ 5 + 32 = {fmt(y, 6)} درجة فهرنهايت."
    else:
        y = x * fac
        if fac >= 1:
            how = f"كل 1 {fu} = {fmt(fac, 6)} {tu}، إذن {x} × {fmt(fac, 6)} = {fmt(y, 6)} {ul}."
        else:
            inv = 1 / fac
            how = f"كل 1 {tu} = {fmt(inv, 6)} {fu}، إذن {x} ÷ {fmt(inv, 6)} = {fmt(y, 6)} {ul}."
    q = [f"حوّل {x} {fu} إلى {tu}.",
         f"قيمة مقدارها {x} {fu}. كم تساوي بوحدة {tu}؟",
         f"في تقرير علمي ذُكرت قيمة {x} {fu}. عبّر عنها بوحدة {tu}."]
    s = how
    return (fu, tu, x), q, s, y, ul, "general", "unit_conversion"


GENS = [g_kin_v, g_kin_s, g_kin_avg, g_newton_F, g_newton_a, g_weight, g_ke, g_pe, g_power,
        g_ohm_I, g_resist, g_epower, g_density, g_pressure, g_wave,
        g_molar_mass, g_moles, g_molarity, g_dilution, g_gas, g_stoich,
        g_punnett, g_dihybrid, g_growth, g_lapse, g_seismic, g_convert]


# ---------------------------------------------------------------- normalisation / leakage
_DIAC = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")
_AR_DIG = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def norm(s):
    s = _DIAC.sub("", s or "").translate(_AR_DIG)
    s = re.sub("[إأآٱ]", "ا", s).replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    s = re.sub(r"[^\w]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def question_of(prompt):
    p = prompt or ""
    if p.startswith(HEAD):
        p = p[len(HEAD):]
    return p.rsplit("\nالحل:", 1)[0].strip()


def make_item(gen_name, tpl_idx, q, steps, val, unit, subject, topic):
    ans = fmt(val)
    return {"task": "reason", "src": "a10_science_gen", "subject": subject, "topic": topic,
            "template": f"{gen_name}#{tpl_idx}", "prompt": HEAD + q + "\nالحل:",
            "completion": f" {steps}\nالجواب: {ans} {unit}\n", "answer": ans, "unit": unit,
            "answer_value": val, "tol_rel": TOL_REL,
            "qhash": hashlib.sha1(q.strip().encode("utf-8")).hexdigest()[:16]}


def load_jsonl(p):
    if not os.path.exists(p):
        return None
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    r = random.Random(SEED)
    train, exam = [], []
    used_train_keys = collections.defaultdict(set)
    seen_norm = set()
    for g in GENS:
        name = g.__name__[2:]
        # train: phrasings 0/1
        tries = 0
        cnt = 0
        while cnt < N_TRAIN_PER_GEN:
            tries += 1
            if tries > 200000:
                raise SystemExit(f"cannot fill train for {name}")
            key, qs, steps, val, unit, subj, top = g(r)
            ti = r.randrange(2)
            n = norm(qs[ti])
            if n in seen_norm:
                continue
            seen_norm.add(n)
            used_train_keys[name].add(key)
            train.append(make_item(name, ti, qs[ti], steps, val, unit, subj, top))
            cnt += 1
        tries = 0
        cnt = 0
        while cnt < N_EXAM_PER_GEN:
            tries += 1
            if tries > 200000:
                raise SystemExit(f"cannot fill exam for {name}")
            key, qs, steps, val, unit, subj, top = g(r)
            if key in used_train_keys[name]:
                continue
            n = norm(qs[2])
            if n in seen_norm:
                continue
            seen_norm.add(n)
            used_train_keys[name + "#exam"].add(key)
            exam.append(make_item(name, 2, qs[2], steps, val, unit, subj, top))
            cnt += 1
    r.shuffle(train)
    r.shuffle(exam)

    # ---- leakage
    def sets(items):
        ps = [x["prompt"] for x in items]
        qs = [question_of(p) for p in ps]
        return {"exact": set(ps), "norm": {norm(p) for p in ps}, "qnorm": {norm(q) for q in qs},
                "qhash": {hashlib.sha1(q.encode("utf-8")).hexdigest()[:16] for q in qs}}

    def overlap(A, B):
        return {k: len(A[k] & B[k]) for k in A}

    tr_s, ex_s = sets(train), sets(exam)
    leak = {"train_vs_exam": overlap(tr_s, ex_s)}
    # parameter-combo overlap per generator (train vs exam)
    leak["train_vs_exam"]["param_combos"] = sum(
        len(used_train_keys[g.__name__[2:]] & used_train_keys[g.__name__[2:] + "#exam"]) for g in GENS)
    ext = {}
    pbm = load_jsonl(os.path.join(T, "packet_b_math.jsonl"))
    pb = load_jsonl(os.path.join(T, "packet_b.jsonl"))
    missing = []
    for nm_, data in (("packet_b_math(A9_exam)", pbm), ("packet_b", pb)):
        if data is None:
            missing.append(nm_)
            continue
        prompts = []
        for x in data:
            for fld in ("prompt", "prompt_bare"):
                if x.get(fld):
                    prompts.append(x[fld])
        B = {"exact": set(prompts), "norm": {norm(p) for p in prompts},
             "qnorm": {norm(question_of(p)) for p in prompts},
             "qhash": {x["qhash"] for x in data if x.get("qhash")} |
                      {hashlib.sha1(question_of(p).encode("utf-8")).hexdigest()[:16] for p in prompts}}
        ext[nm_] = {"n": len(data), "train": overlap(tr_s, B), "exam": overlap(ex_s, B)}
    leak.update(ext)
    total = sum(leak["train_vs_exam"].values()) + sum(
        sum(v[s].values()) for v in ext.values() for s in ("train", "exam"))
    if missing:
        total = -1  # cannot certify

    with open(os.path.join(T, "packet_a10_train.jsonl"), "w", encoding="utf-8") as f:
        for x in train: f.write(json.dumps(x, ensure_ascii=False) + "\n")
    with open(os.path.join(T, "packet_a10_exam.jsonl"), "w", encoding="utf-8") as f:
        for x in exam: f.write(json.dumps(x, ensure_ascii=False) + "\n")

    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
    lines = [
        f"A10 science | seed={SEED} | tol_rel={TOL_REL} | generators={len(GENS)} | phrasings train=0,1 exam=2",
        f"train={len(train)} exam={len(exam)}",
        f"train by subject {dict(sorted(collections.Counter(x['subject'] for x in train).items()))}",
        f"exam  by subject {dict(sorted(collections.Counter(x['subject'] for x in exam).items()))}",
        f"train by topic {dict(sorted(collections.Counter(x['topic'] for x in train).items()))}",
        f"exam  by topic {dict(sorted(collections.Counter(x['topic'] for x in exam).items()))}",
        f"leakage train_vs_exam {leak['train_vs_exam']}",
    ]
    for k, v in ext.items():
        lines.append(f"leakage vs {k} (n={v['n']}): train {v['train']} | exam {v['exam']}")
    if missing:
        lines.append(f"MISSING reference files: {missing}")
    lines.append(f"LEAKAGE={total}")
    lines.append(f"packet_a10_train sha256[:16]={sha(os.path.join(T, 'packet_a10_train.jsonl'))} | "
                 f"packet_a10_exam sha256[:16]={sha(os.path.join(T, 'packet_a10_exam.jsonl'))}")
    msg = "\n".join(lines)
    print(msg)
    open(os.path.join(T, "packet_a10_build.log"), "w", encoding="utf-8").write(msg + "\n")


if __name__ == "__main__":
    main()
