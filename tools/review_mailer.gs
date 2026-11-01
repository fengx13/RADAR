/**
 * RADAR clinician review: receive results from docs/review/index.html,
 * keep every submission in this spreadsheet and email a copy to the team.
 *
 * Set up (once, about 5 minutes, signed in with your UMN Google account):
 * 1. Create a Google Sheet named "RADAR review results". Extensions > Apps Script.
 * 2. Replace the code with this file. Save.
 * 3. Deploy > New deployment > type "Web app".
 *    Execute as: Me.  Who has access: Anyone.  Deploy, authorize, copy the Web app URL.
 * 4. Paste that URL into SEND_URL near the top of the script in docs/review/index.html, commit and push.
 */
const TEAM_EMAIL = "jin00330@umn.edu";

const SHEET_ID = "19J6xijyiNoPI5qwC7lVNxqYGCETuKumBaO_XHumJDmE"; // the "RADAR review results" sheet

function resultsSheet_() {
  return SpreadsheetApp.openById(SHEET_ID).getSheets()[0];
}

/** Run once from the editor (select testSetup, press Run) to grant permissions and check that the email arrives. */
function testSetup() {
  doPost({ postData: { contents: JSON.stringify({ reviewer: { name: "Setup test" }, case: "test", rating: 5, comment: "test row, delete me" }) } });
}

function doPost(e) {
  const raw = e && e.postData ? e.postData.contents : "";
  let d = {};
  try { d = JSON.parse(raw); } catch (err) { d = { parse_error: String(err) }; }
  const sheet = resultsSheet_();
  if (sheet.getLastRow() === 0) sheet.appendRow(["received", "reviewer", "case", "finished", "rating", "comment", "json"]);
  const name = (d.reviewer && d.reviewer.name) || "unknown";
  sheet.appendRow([new Date(), name, d.case || "", d.submittedAt || "", d.rating || "", d.comment || "", raw]);
  const file = Utilities.newBlob(raw, "application/json", "RADAR_review_" + name.replace(/[^A-Za-z0-9]+/g, "_") + ".json");
  MailApp.sendEmail({ to: TEAM_EMAIL, subject: "RADAR review - " + name, body: "New RADAR clinician review from " + name + ". The full answers are attached and saved in the sheet.", attachments: [file] });
  return ContentService.createTextOutput("ok");
}
