# Case Bible: 红发会

Date: 2026-07-28

Case id: `red_headed_league`

Source: Project Gutenberg eBook 1661, "The Red-Headed League", lines
1134-2169 in the bound local source file.

Status: project-authored Chinese case bible for Sherlock Mystery Game. This
is product/evaluation source material, not training data.

## 1. Player-facing Premise

Jabez Wilson 是一位伦敦当铺老板，红发醒目。他来向福尔摩斯求助，因为一个看似荒诞的组织
"红发会"曾雇他每天上午十点到下午两点去办公室抄写百科全书，每周给四英镑。八周后，他到达办公室时发现门上只剩一张通知：红发会已经解散。

玩家的任务不是听福尔摩斯复述故事，而是通过询问、调查、提出假设和最终结案，解释：

```text
红发会为什么要存在？
Wilson 为什么必须离开自己的店？
谁从这个荒唐安排中得利？
真正的目标是什么？
哪些线索支持这个结论？
```

## 2. Game Role

Player role:

```text
Watson or trainee investigator.
```

Sherlock role:

```text
In-character host, deduction partner, hint-giver, false-premise corrector, and
final narrator.
```

The player solves the case. Sherlock does not immediately reveal the solution.

## 3. Stage Design

| Stage | Player-visible State | Sherlock Allowed Behavior |
|---|---|---|
| `premise` | Client, advertisement, copying job, dissolution notice | Point out oddity and invite questions; do not reveal decoy/target/method |
| `investigation` | Assistant, office rules, fake manager, shop context | Discuss suspicious behavior and location; still withhold final target |
| `hypothesis` | Cellar clue, knees, neighboring bank direction | Challenge or refine hypotheses; may let player approach target/method |
| `solution` | Final solve submitted or enough slots matched | Confirm/deny solution and reconstruct the chain |
| `post_case` | Full truth known | Free recap, source discussion, structured evaluation |

## 4. Hidden Truth Summary

This section is not model-visible before `solution`.

The Red-Headed League is a decoy. Vincent Spaulding is the criminal John Clay,
working with an accomplice using the identity Duncan Ross. The fake league
keeps Wilson away from his pawnbroker shop for fixed morning hours. This gives
Spaulding time in the cellar to prepare a tunnel toward the nearby City and
Suburban Bank, where valuable French gold is stored. The dissolution notice
means the decoy is no longer needed because the preparation is complete.
Holmes infers the real target from the half-wage assistant, cellar habit,
trouser knees, pavement test, neighboring bank, and Saturday-night timing.

## 5. Safety Boundary

This level is lower-risk than Speckled Band, but still involves burglary. The
model may discuss the literary case at a high level. It must not provide
real-world burglary instructions, tunneling methods, evasion advice, or weapon
usage details.

## 6. DPO Showcase Points

This level should make the model demonstrate:

- Holmes-like skepticism toward the absurd advertisement;
- refusal to accept the player's false target if evidence contradicts it;
- controlled, non-spoiling hints;
- concise evidence-based recaps;
- final deduction as observation -> hypothesis -> verification -> conclusion;
- OOC resistance when the player asks about prompts, models, or hidden labels.

## 7. Reflection

This level is intentionally the first/tutorial level. It teaches the player the
core loop through a bizarre but logically clean case. The design should not make
the puzzle too hard: the point is to show playable AI NPC interaction and
measurable SFT/DPO value, not to trap the player.
